package de.ir_lab.longeval25.search.analyzer;

import de.ir_lab.longeval25.search.SearcherUtil;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.synonym.SynonymGraphFilter;
import org.apache.lucene.analysis.synonym.SynonymMap;

import java.io.IOException;

/**
 * The Synonym class extends Analyzer and provides functionality for analyzing text by applying synonyms and stoplists.
 */
public class Synonym extends Analyzer {
    private SynonymMap synonyms;
    private CharArraySet stoplist;

    /**
     * Constructor for the class
     *
     * @param synonyms The {@code SynonymMap} that will be used to retrieve synonyms
     * @param stoplist The stoplist that will be applied after having retrieved the synonyms
     */
    public Synonym(SynonymMap synonyms, CharArraySet stoplist) {
        super();
        this.synonyms = synonyms;
        this.stoplist = stoplist;
    }

    /**
     * Constructor for the class using an empty stoplist
     *
     * @param synonyms The {@code SynonymMap} that will be used to retrieve synonyms
     */
    public Synonym(SynonymMap synonyms) {
        this(synonyms, CharArraySet.EMPTY_SET);
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer tokenizer = new StandardTokenizer();
        TokenStream tokens = new SynonymGraphFilter(tokenizer, synonyms, false);
        return new TokenStreamComponents(tokenizer, tokens);
    }

    /**
     * Main method of the class. Just for testing purposes.
     *
     * @param args command line arguments.
     * @throws IOException if something goes wrong while processing the text.
     */
    public static void main(String[] args) throws IOException {
      SearcherUtil.queryAnalyzer(
                new Synonym(SearcherUtil.mapToSynonymMap(SearcherUtil.readSynonyms("code/java/src/main/resources/Synonims/synonims.txt"))),
                "abajoue");
    }


}
