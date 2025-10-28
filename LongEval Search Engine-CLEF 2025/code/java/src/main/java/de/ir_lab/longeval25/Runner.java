/*
 * Copyright 2021-2022 University of Padua, Italy
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package de.ir_lab.longeval25;

import de.ir_lab.longeval25.analyzer.FrenchAnalyzer;
import de.ir_lab.longeval25.index.DirectoryIndexer;
import de.ir_lab.longeval25.parse.TrecParser;
import de.ir_lab.longeval25.search.Searcher;
import de.ir_lab.longeval25.utility.ConfigManager;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.Similarity;

/**
 * Introductory example on how to use <a href="https://lucene.apache.org/" target="_blank">Apache Lucene</a> to index
 * and search the TIPSTER corpus.
 *
 * @author Searchill
 * @version 1.0
 * @since 1.0
 */
public class Runner {

    private static final ConfigManager config = ConfigManager.getInstance();


    /**
     * Main method of the class.
     *
     * @param args command line arguments. If provided, {@code args[0]} contains the path the the index directory;
     *             {@code args[1]} contains the path to the run file.
     * @throws Exception if something goes wrong while indexing and searching.
     */
    public static void main(String[] args) throws Exception {

        final int expectedDocs = 17382;

        final Analyzer frenchAnalyzer = new FrenchAnalyzer();
        final Similarity sim = new BM25Similarity();


        // indexing
        final DirectoryIndexer i = new DirectoryIndexer(frenchAnalyzer, sim, 256,
                config.getString("indexPath"),
                config.getString("collectionPath"),
                "trec", "ISO-8859-1",
                expectedDocs, TrecParser.class);
        i.index();

        // searching
        Searcher s = new Searcher(frenchAnalyzer, new BM25Similarity(),
                config.getString("indexPath"),
                config.getString("topics"),
                500,
                config.getString("runID"),
                config.getString("runPath"),
                config.getInt("maxDocsRetrieved"));
        s.search();

    }

}
