/*
 *  Copyright 2021-2022 University of Padua, Italy
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package de.ir_lab.longeval25.search;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceTokenizerFactory;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.benchmark.quality.QualityQuery;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.FuzzyQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.PhraseQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.FSDirectory;

import de.ir_lab.longeval25.analyzer.FrenchAnalyzer;
import de.ir_lab.longeval25.parse.ParsedDocument;
import de.ir_lab.longeval25.search.analyzer.Ngrams;
import de.ir_lab.longeval25.search.analyzer.Synonym;
import de.ir_lab.longeval25.utility.ConfigManager;

import static de.ir_lab.longeval25.search.SearcherUtil.*;

/**
 * Searches a document collection.
 *
 * @author Searchill
 * @version 1.00
 * @since 1.00
 */
public class Searcher {

    /**
     * The fields of the typical TREC topics.
     *
     * @author Searchill
     * @version 1.00
     * @since 1.00
     */
    private static final class TOPIC_FIELDS {

        /**
         * The title of a topic.
         */
        public static final String TITLE = "title";

        /**
         * The description of a topic.
         */
        public static final String DESCRIPTION = "description";

        /**
         * The narrative of a topic.
         */
        public static final String NARRATIVE = "narrative";
    }

    private static final ConfigManager config = ConfigManager.getInstance();

    /**
     * The identifier of the run
     */
    private final String runID;

    /**
     * The run to be written
     */
    private final PrintWriter run;

    /**
     * The index reader
     */
    private final IndexReader reader;

    /**
     * The index searcher.
     */
    private final IndexSearcher searcher;

    /**
     * The topics to be searched
     */
    private final QualityQuery[] topics;

    /**
     * The query parser
     */
    private final QueryParser qp;

    /**
     * The maximum number of documents to retrieve
     */
    private final int maxDocsRetrieved;

    /**
     * The total elapsed time.
     */
    private long elapsedTime = Long.MIN_VALUE;


    /**
     * Creates a new searcher.
     *
     * @param analyzer         the {@code Analyzer} to be used.
     * @param similarity       the {@code Similarity} to be used.
     * @param indexPath        the directory where containing the index to be searched.
     * @param topicsFile       the file containing the topics to search for.
     * @param expectedTopics   the total number of topics expected to be searched.
     * @param runID            the identifier of the run to be created.
     * @param runPath          the path where to store the run.
     * @param maxDocsRetrieved the maximum number of documents to be retrieved.
     * @throws NullPointerException     if any of the parameters is {@code null}.
     * @throws IllegalArgumentException if any of the parameters assumes invalid values.
     */
    public Searcher(final Analyzer analyzer, final Similarity similarity, final String indexPath,
                    final String topicsFile, final int expectedTopics, final String runID, final String runPath,
                    final int maxDocsRetrieved) {

        if (analyzer == null) {
            throw new NullPointerException("Analyzer cannot be null.");
        }

        if (similarity == null) {
            throw new NullPointerException("Similarity cannot be null.");
        }

        if (indexPath == null) {
            throw new NullPointerException("Index path cannot be null.");
        }

        if (indexPath.isEmpty()) {
            throw new IllegalArgumentException("Index path cannot be empty.");
        }

        final Path indexDir = Paths.get(indexPath);
        if (!Files.isReadable(indexDir)) {
            throw new IllegalArgumentException(
                    String.format("Index directory %s cannot be read.", indexDir.toAbsolutePath().toString()));
        }

        if (!Files.isDirectory(indexDir)) {
            throw new IllegalArgumentException(String.format("%s expected to be a directory where to search the index.",
                    indexDir.toAbsolutePath().toString()));
        }

        try {
            reader = DirectoryReader.open(FSDirectory.open(indexDir));
        } catch (IOException e) {
            throw new IllegalArgumentException(String.format("Unable to create the index reader for directory %s: %s.",
                    indexDir.toAbsolutePath().toString(), e.getMessage()), e);
        }

        searcher = new IndexSearcher(reader);
        searcher.setSimilarity(similarity);

        if (topicsFile == null) {
            throw new NullPointerException("Topics file cannot be null.");
        }

        if (topicsFile.isEmpty()) {
            throw new IllegalArgumentException("Topics file cannot be empty.");
        }

        try {
            BufferedReader in = Files.newBufferedReader(Paths.get(topicsFile), StandardCharsets.UTF_8);
            topics = SearcherUtil.readTabDelimitedTopics(in);
            in.close();
        } catch (IOException e) {
            throw new IllegalArgumentException(
                    String.format("Unable to process topic file %s: %s.", topicsFile, e.getMessage()), e);
        }

        if (expectedTopics <= 0) {
            throw new IllegalArgumentException(
                    "The expected number of topics to be searched cannot be less than or equal to zero.");
        }

        if (topics.length != expectedTopics) {
            System.out.printf("Expected to search for %s topics; %s topics found instead.", expectedTopics,
                    topics.length);
        }

        qp = new QueryParser(ParsedDocument.FIELDS.BODY, analyzer);

        if (runID == null) {
            throw new NullPointerException("Run identifier cannot be null.");
        }

        if (runID.isEmpty()) {
            throw new IllegalArgumentException("Run identifier cannot be empty.");
        }

        this.runID = runID;


        if (runPath == null) {
            throw new NullPointerException("Run path cannot be null.");
        }

        if (runPath.isEmpty()) {
            throw new IllegalArgumentException("Run path cannot be empty.");
        }

        final Path runDir = Paths.get(runPath);
        if (!Files.isWritable(runDir)) {
            throw new IllegalArgumentException(
                    String.format("Run directory %s cannot be written.", runDir.toAbsolutePath().toString()));
        }

        if (!Files.isDirectory(runDir)) {
            throw new IllegalArgumentException(String.format("%s expected to be a directory where to write the run.",
                    runDir.toAbsolutePath().toString()));
        }

        Path runFile = runDir.resolve(runID + ".txt");
        try {
            run = new PrintWriter(Files.newBufferedWriter(runFile, StandardCharsets.UTF_8, StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE));
        } catch (IOException e) {
            throw new IllegalArgumentException(
                    String.format("Unable to open run file %s: %s.", runFile.toAbsolutePath(), e.getMessage()), e);
        }

        if (maxDocsRetrieved <= 0) {
            throw new IllegalArgumentException(
                    "The maximum number of documents to be retrieved cannot be less than or equal to zero.");
        }

        this.maxDocsRetrieved = maxDocsRetrieved;
    }

    /**
     * Returns the total elapsed time.
     *
     * @return the total elapsed time.
     */
    public long getElapsedTime() {
        return elapsedTime;
    }

    /**
     * /** Searches for the specified topics.
     *
     * @throws IOException    if something goes wrong while searching.
     * @throws ParseException if something goes wrong while parsing topics.
     */
    public void search() throws IOException, ParseException {

        System.out.printf("%n#### Start searching ####%n");

        // the start time of the searching
        final long start = System.currentTimeMillis();

        final Set<String> idField = new HashSet<>();
        idField.add(ParsedDocument.FIELDS.ID);

        BooleanQuery.Builder booleanQuery;
        Query query;
        TopDocs docs;
        ScoreDoc[] scoreDocs;
        String docID;
        int topicsCount = 0;
        int rerankCount = 1;
        boolean isBig = false;

        IndexSearcher.setMaxClauseCount(10000);

        Analyzer analyzer = new FrenchAnalyzer();

        try {
            for (QualityQuery topic : topics) {
                isBig = false;
                topicsCount++;

                System.out.printf("Searching for topic %d of %d: id=%s,title=%s%n",
                        topicsCount, topics.length, topic.getQueryID(), topic.getValue(TOPIC_FIELDS.TITLE));

                booleanQuery = new BooleanQuery.Builder();

                String queryTitle = topic.getValue(TOPIC_FIELDS.TITLE);
                List<String> tokens = queryAnalyzer(analyzer, queryTitle);

                int maxTokens = config.getInt("maxTokens");
                if (tokens.size() > maxTokens) {
                    isBig = true;
                    try{
                        for(String token : tokens){
                            booleanQuery.add(new BoostQuery(qp.parse(token), 0.1f), BooleanClause.Occur.SHOULD);
                        }
                        System.err.println("Truncated tokens to " + maxTokens + " tokens.");
                    } catch (Exception e){
                        System.err.println("Skipping truncated query for ID: " + topic.getQueryID());
                    }
                }

                String analyzedQuery = String.join(" ", tokens);
                analyzedQuery = parsQuery(analyzedQuery);

                if (analyzedQuery.trim().isEmpty()) {
                    System.err.println("Skipping empty analyzed query for ID: " + topic.getQueryID());
                    continue;
                }

                try {
                    //add to the boolean query the terms of the original query
                    booleanQuery.add(new BoostQuery(qp.parse(analyzedQuery), 1f), BooleanClause.Occur.MUST);

                    if (Boolean.TRUE.equals(config.getBool("correctSpelling")) && !isBig) {
                        String correctedText = correctText(queryTitle, detectLanguage(queryTitle));
                        booleanQuery.add(new BoostQuery(new TermQuery(new Term("body", correctedText)), 0.5f), BooleanClause.Occur.SHOULD);
                    }

                    //query expansion for no spaced query
                    if (Boolean.TRUE.equals(config.getBool("dictionary")) && !isBig) {
                        List<String> dict = createDictionary(config.getString("synonymsFile"));
                        for (String word : dict)
                            if (analyzedQuery.contains(word))
                                booleanQuery.add(new BoostQuery(qp.parse(word), 0.1f), BooleanClause.Occur.SHOULD);
                    }

                    //check query on the start
                    if (Boolean.TRUE.equals(config.getBool("start")) && !isBig)
                        booleanQuery.add(new BoostQuery(new TermQuery(new Term("start", analyzedQuery)), 1.5f), BooleanClause.Occur.SHOULD);

                    //check query on the highlight
                    if (Boolean.TRUE.equals(config.getBool("highlight")) && !isBig)
                        booleanQuery.add(new BoostQuery(new TermQuery(new Term("highlight", analyzedQuery)), 1.2f), BooleanClause.Occur.SHOULD);

                    //fuzzy search with FuzzyQuery
                    if (Boolean.TRUE.equals(config.getBool("fuzzy")) && !isBig) {
                        booleanQuery.add(new BoostQuery(new FuzzyQuery(new Term("body", analyzedQuery), 2), 0.8f),
                                BooleanClause.Occur.SHOULD);
                        for (String token : tokens) {
                            booleanQuery.add(new BoostQuery(new FuzzyQuery(new Term("body", token), 2), 0.2f),
                                    BooleanClause.Occur.SHOULD);
                        }
                    }

                    //phrase query
                    if (Boolean.TRUE.equals(config.getBool("phrase")) && tokens.size() > 1 && !isBig) {
                        PhraseQuery.Builder builder = new PhraseQuery.Builder();
                        int maxTerms = Math.min(5, tokens.size());
                        int position = 0;

                        for (int i = 0; i < maxTerms; i++) {
                            String token = tokens.get(i);
                            if (token != null && !token.isEmpty()) {
                                builder.add(new Term("body", token), position++);
                            }
                        }

                        builder.setSlop(5);
                        PhraseQuery phraseQuery = builder.build();
                        booleanQuery.add(new BoostQuery(phraseQuery, 1f), BooleanClause.Occur.SHOULD);
                    }

                    // ======= QUERY EXPANSION =======

                    //query expansion with nGrams
                    if (tokens.size() > 2 && Boolean.TRUE.equals(config.getBool("nGrams")) && !isBig) {
                        List<String> nGramsTokens = queryAnalyzer(new Ngrams(), analyzedQuery);
                        nGramsTokens.removeIf(s -> !s.contains(" "));
                        for (String nGramToken : nGramsTokens)
                            booleanQuery.add(new BoostQuery(new TermQuery(new Term("body", nGramToken)), 0.3f),
                                    BooleanClause.Occur.SHOULD);
                    }

                    //query expansion with synonyms
                    if (Boolean.TRUE.equals(config.getBool("synonyms")) && !isBig) {
                        List<String> synonyms = SearcherUtil.queryAnalyzer(
                                new Synonym(SearcherUtil.mapToSynonymMap(SearcherUtil.readSynonyms(config.getString("synonymsFile")))),
                                queryTitle);
                        for (String synonym : synonyms) {
                            String[] split = synonym.split(",");
                            if (split.length > 1) {
                                List<String> tmpSynonyms = new ArrayList<>(Arrays.asList(split));
                                for (String tmpSynonym : tmpSynonyms)
                                    booleanQuery.add(new BoostQuery(new TermQuery(new Term("body", tmpSynonym)), 1f/tmpSynonyms.size()), BooleanClause.Occur.SHOULD);
                            }
                        }
                    }

                    //query expansion with LLM
                    if (Boolean.TRUE.equals(config.getBool("useLLMExpansion")) && !isBig) {
                        String[] relatedTerms = getRelatedTermsFromLLM(queryTitle, config.getOpenApiKey());

                        for (String relatedTerm : relatedTerms)
                            booleanQuery.add(new BoostQuery(new TermQuery(new Term("body", relatedTerm)), 0.4f), BooleanClause.Occur.SHOULD);

                    }

                    //query expansion word2vec
                    if (Boolean.TRUE.equals(config.getBool("useWord2Vec")) && !isBig) {

                        String[] relatedTerms = getRelatedTermsWord2Vec(SearcherUtil.queryAnalyzer(
                                CustomAnalyzer.builder().withTokenizer(WhitespaceTokenizerFactory.class).build(),
                                queryTitle));

                        for (String relatedTerm : relatedTerms)
                            booleanQuery.add(new BoostQuery(new TermQuery(new Term("body", relatedTerm)), 0.1f), BooleanClause.Occur.SHOULD);

                    }


                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                query = booleanQuery.build();
                docs = searcher.search(query, Boolean.TRUE.equals(config.getBool("prf")) ? config.getInt("numOfDocsToRetrieveForPrf") : maxDocsRetrieved);
                scoreDocs = docs.scoreDocs;

                // ======= PRF =======
                if (Boolean.TRUE.equals(config.getBool("prf")) && !isBig) {

                    String[] prfTerms = getRelatedTermsFromPrf(scoreDocs, reader);
                    for (String relatedTerm : prfTerms) {
                        booleanQuery.add(new BoostQuery(new TermQuery(new Term("body", relatedTerm)), 0.5f), BooleanClause.Occur.SHOULD);
                    }
                    query = booleanQuery.build();
                    docs = searcher.search(query, maxDocsRetrieved);
                    scoreDocs = docs.scoreDocs;
                }


                // ======= RERANKER =======
                if(Boolean.TRUE.equals(config.getBool("reRank_heuristic"))){
                    for (ScoreDoc scoreDoc : scoreDocs) {
                        Document docum = searcher.doc(scoreDoc.doc);
                        HeuristicReranker.rerankScore(docum, tokens, scoreDoc, reader, scoreDoc.doc);
                    }
                }
                float minScore = Float.MAX_VALUE;
                float maxScore = Float.MIN_VALUE;
                for (ScoreDoc scoreDoc : scoreDocs) {
                    if (scoreDoc.score < minScore) {
                        minScore = scoreDoc.score;
                    }
                    if (scoreDoc.score > maxScore) {
                        maxScore = scoreDoc.score;
                    }
                }
                if (maxScore != minScore) {
                    for (ScoreDoc scoreDoc : scoreDocs) {
                        scoreDoc.score = (scoreDoc.score - minScore) / (maxScore - minScore);
                    }
                }

                // ======= RERANKER =======
                if (Boolean.TRUE.equals(config.getBool("reRank")) && !isBig) {

                    Map<String, List<String>> docMap = SearcherUtil.retrieveDocuments(scoreDocs, reader, queryTitle);
                    if (!docMap.isEmpty()) {
                        if(Boolean.TRUE.equals(config.getBool("reRank"))){
                        String response = SearcherUtil.reRank(queryTitle, docMap.get(queryTitle), config.getInt("numOfDocsToRerank"), "key");
                        Map<Integer, Float> newScores = SearcherUtil.jsonHandler(response);
                        SearcherUtil.fixScores(newScores, scoreDocs, config.getInt("numOfDocsToRerank"), 0.6f);
                        }
                        rerankCount++;

                    } else {
                        System.err.printf("No documents found for the query: %s%n", queryTitle);
                    }
                }
                
                // Write the re-ranked results to the file
                for (int i = 0, n = scoreDocs.length; i < n; i++) {
                    docID = reader.document(scoreDocs[i].doc, idField).get(ParsedDocument.FIELDS.ID);

                    // Remove the prefix "doc" from the document ID if it exists
                    if (docID.startsWith("doc")) {
                        docID = docID.substring(3); // Rimuove i primi 3 caratteri ("doc")
                    }

                    run.printf(Locale.ENGLISH, "%s Q0 %s %d %d %s%n", topic.getQueryID(), docID, i, (int) scoreDocs[i].score, runID);
                }

                run.flush();
            }
        } finally {
            run.close();
            reader.close();
        }

        elapsedTime = System.currentTimeMillis() - start;

        System.out.printf("%d topic(s) searched in %d seconds.%n", topics.length, elapsedTime / 1000);

        System.out.printf("#### Searching complete ####%n");

    }


    /**
     * Main method of the class. Just for testing purposes.
     *
     * @param args command line arguments.
     * @throws Exception if something goes wrong while indexing.
     */
    public static void main(String[] args) throws Exception {

        final Analyzer frenchAnalyzer = new FrenchAnalyzer();

        Searcher s = new Searcher(frenchAnalyzer, new BM25Similarity(),
                config.getString("indexPath"),
                config.getString("topics"),
                3019,
                config.getString("runID"),
                config.getString("runPath"),
                config.getInt("maxDocsRetrieved"));

        s.search();
    }

}
