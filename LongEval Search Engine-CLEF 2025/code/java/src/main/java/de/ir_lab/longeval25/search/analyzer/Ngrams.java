package de.ir_lab.longeval25.search.analyzer;

import de.ir_lab.longeval25.search.SearcherUtil;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.shingle.ShingleFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;

import java.io.IOException;

/**
 * The Ngrams class extends Lucene's Analyzer class and provides functionality
 * for generating n-grams from text input.
 */
public class Ngrams extends Analyzer {

    /**
     * Constructs a new Ngrams analyzer.
     */
    public Ngrams(){
        super();
    }

    /**
     * Creates token stream components for the given field name.
     *
     * @param fieldName The name of the field for which token stream components are created.
     * @return TokenStreamComponents containing the tokenizer and token filter chain.
     */
    @Override
    protected TokenStreamComponents createComponents(String fieldName){
        Tokenizer source = new StandardTokenizer();
        return  new TokenStreamComponents(source, new ShingleFilter(source));
    }

    /**
     * Main method for testing purposes.
     *
     * @param args Command line arguments.
     * @throws IOException If an IO error occurs while processing text.
     */
    public static void main(String[] args) throws IOException {
        final String text = "Comment allez-vous";

        System.out.println(SearcherUtil.queryAnalyzer(new Ngrams(), text));
    }

}

