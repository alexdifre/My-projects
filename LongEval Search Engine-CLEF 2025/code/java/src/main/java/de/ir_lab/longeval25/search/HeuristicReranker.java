package de.ir_lab.longeval25.search;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.util.BytesRef;

import de.ir_lab.longeval25.analyzer.FrenchAnalyzer;
import static de.ir_lab.longeval25.search.SearcherUtil.queryAnalyzer;


public class HeuristicReranker {
    private static final String FIELD_NAME = "content"; 
    private static final Analyzer analyzer = new FrenchAnalyzer();

    public static void rerankScore(Document doc, List<String> queryTerms, ScoreDoc scoreDoc, IndexReader reader, int docId) {
        try {

            String body = doc.get(FIELD_NAME);  
            List<String> tokens = queryAnalyzer(analyzer, body);
            
            Map<String, Long> docTerms = getDocumentTerms(reader, docId);
            
            // 3. Calcola metriche
            long docLength = tokens.size();
            int queryLength = queryTerms.size();
            int VocabLength = docTerms.size();

            
            // 4. Applica penalità
            lengthPenalty( scoreDoc,  docLength);
            vocabularyPenalty(scoreDoc, docLength, VocabLength);
            spamTFpenalty( scoreDoc, docTerms, queryTerms, docLength);
            repetitionPenalty(scoreDoc, docTerms, docLength, VocabLength);
            float richness = (float) VocabLength / docLength;

            //  QUERY HEURISTICS
                //Query breve  premia documenti lunghi o vari
                if (queryLength <= 2) {
                    
                    if (docLength  > 100 && richness > 0.4) {
                        scoreDoc.score *= 1.1f; // documento vario e lungo
                    }
                }
                if (queryLength >= 12) {

                    // uery con parole varie premia documenti con varietà
                    if ((float) queryLength / queryLength > 0.8) {
                        if (richness > 0.5) {
                            scoreDoc.score *= 1.1f;
                        }
                    }
                }
            
        } catch (IOException e) {
            System.err.println("Error reranking doc " + docId + ": " + e.getMessage());
        }
    }
    
    
private static Map<String, Long> getDocumentTerms(IndexReader reader, int docId) throws IOException {
    Terms terms = reader.getTermVector(docId, FIELD_NAME);
    if (terms == null) return Collections.emptyMap();
    
    Map<String, Long> termTfMap = new HashMap<>();
    TermsEnum termsEnum = terms.iterator();
    BytesRef term;
    
    while ((term = termsEnum.next()) != null) {
        String termString = term.utf8ToString().toLowerCase();
        
        PostingsEnum postings = termsEnum.postings(null);
        if (postings != null && postings.nextDoc() != PostingsEnum.NO_MORE_DOCS) {
            long tf = postings.freq();
            termTfMap.put(termString, tf);
        }
    }
    
    return termTfMap;
}


    // Penalizza solo documenti troppo corti o troppo lunghi, ma più gradualmente
    private static void lengthPenalty(ScoreDoc scoreDoc, long length) {
        int MIN_LEN = 150, MAX_LEN = 15000;
        if (length < MIN_LEN) {
            float factor = 0.8f + (float) length / MIN_LEN * 0.2f; // da 0.8 a 1.0
            scoreDoc.score *= factor;
        } else if (length > MAX_LEN) {
            float factor = 1.0f - (float) (length - MAX_LEN) / (MAX_LEN * 2); // scende lentamente
            scoreDoc.score *= Math.max(0.85f, factor); // non scende sotto 0.85
        }
    }

    private static void vocabularyPenalty(ScoreDoc scoreDoc, long totalTerms, int uniqueTerms) {
        // Calcola ilnumero totale di termini nel documento
        if (totalTerms == 0) return;
        
        float richness = (float) uniqueTerms / totalTerms;
        
        // Applica penalità/bonus con soglie più realistiche
        if (richness < 0.05f) {
            // Vocabolario molto povero (meno del 5% di varietà)
            scoreDoc.score *= 0.75f; // -25%
        } else if (richness < 0.08f) {
            // Vocabolario povero (meno dell'8% di varietà)
            scoreDoc.score *= 0.85f; // -15%
        } else if (richness < 0.12f) {
            // Vocabolario medio-basso (meno del 12% di varietà)
            scoreDoc.score *= 0.93f; // -7%
        } else if (richness >= 0.20f) {
            // Vocabolario molto ricco (20%+ di varietà)
            scoreDoc.score *= 1.10f; // +10%
        } else if (richness >= 0.15f) {
            // Vocabolario ricco (15%+ di varietà)
            scoreDoc.score *= 1.05f; // +5%
        }
    }
    
    /**
     * Penalizza documenti che abusano dei termini della query (keyword stuffing)
     * Versione unificata e corretta con soglie più realistiche
     */
    private static void spamTFpenalty(ScoreDoc scoreDoc, Map<String, Long> docTermFreqs, List<String> queryTerms, long totalTerms) {
        if (queryTerms == null || queryTerms.isEmpty()) return;
        
        
        if (totalTerms == 0) return;
        
        // Calcola la frequenza totale dei termini della query nel documento
        long totalQueryTermFreq = 0;
        for (String queryTerm : queryTerms) {
            String normalizedTerm = queryTerm.toLowerCase().trim();
            totalQueryTermFreq += docTermFreqs.getOrDefault(normalizedTerm, 0L);
        }
        
        // Calcola la densità dei termini della query
        float density = (float) totalQueryTermFreq / totalTerms;
        
        // Applica penalità progressiva con soglie più aggressive
        if (density > 0.15f) {
            // Densità molto alta (> 15%) - probabile spam aggressivo
            float excessDensity = density - 0.15f;
            float penaltyPercentage = Math.min(85f, excessDensity * 400f); // Penalità più aggressiva
            float penaltyFactor = 1f - (penaltyPercentage / 100f);
            scoreDoc.score *= penaltyFactor;
        } else if (density > 0.10f) {
            // Densità alta (10-15%) - possibile keyword stuffing
            float excessDensity = density - 0.10f;
            float penaltyPercentage = Math.min(40f, excessDensity * 200f);
            float penaltyFactor = 1f - (penaltyPercentage / 100f);
            scoreDoc.score *= penaltyFactor;
        } else if (density > 0.08f) {
            // Densità medio-alta (8-10%) - penalità leggera
            scoreDoc.score *= 0.95f; // -5%
        }
    }
    
    private static void repetitionPenalty(ScoreDoc scoreDoc, Map<String, Long> docTermFreqs, long totalTerms, int uniqueTerms) {
        if (docTermFreqs == null || docTermFreqs.isEmpty()) return;
        
        long maxFreq = docTermFreqs.values().stream().mapToLong(Long::longValue).max().orElse(0);
        
        if (totalTerms == 0) return;
        
        // Calcola indicatori di ripetizione
        float richness = (float) uniqueTerms / totalTerms;
        float maxTermRatio = (float) maxFreq / totalTerms;
        
        // Calcola punteggio combinato di ripetizione
        float repetitionScore = (0.7f * maxTermRatio) + (0.3f * (1 - richness));
        
        // Applica penalità progressiva con soglie più sensibili
        if (repetitionScore > 0.25f) {
            // Ripetizione molto alta - probabile spam o contenuto generato automaticamente
            float severity = Math.min(1.0f, (repetitionScore - 0.25f) / 0.75f);
            float penaltyFactor = 1.0f - (severity * 0.70f); // Fino al -70%
            scoreDoc.score *= penaltyFactor;
        } else if (repetitionScore > 0.18f) {
            // Ripetizione alta - penalità media
            float severity = (repetitionScore - 0.18f) / 0.07f;
            float penaltyFactor = 1.0f - (severity * 0.30f); // Fino al -30%
            scoreDoc.score *= penaltyFactor;
        } else if (repetitionScore > 0.12f) {
            // Ripetizione medio-alta - penalità leggera
            scoreDoc.score *= 0.90f; // -10%
        }
    }
}
    

