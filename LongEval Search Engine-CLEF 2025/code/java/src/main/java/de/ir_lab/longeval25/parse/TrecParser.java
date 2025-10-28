/*
 *  Copyright 2017-2022 University of Padua, Italy
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

package de.ir_lab.longeval25.parse;

import com.vdurmont.emoji.EmojiParser;
import de.ir_lab.longeval25.utility.ConfigManager;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 *
 * @author Searchill
 * @version 1.00
 * @since 1.00
 */
public class TrecParser extends DocumentParser {

    /**
     * The size of the buffer for the body element.
     */
    private static final int BODY_SIZE = 1024 * 8;

    /**
     * The currently parsed document
     */
    private ParsedDocument document = null;

    /**
     * Regex that finds <DOCNO> tag
     */
    private static final Pattern DOCNO_TAG = Pattern.compile("<DOCNO>\\s*(\\S+)\\s*<");

    /**
     * Regex that finds <DOCID> tag
     */
    private static final Pattern DOC_ID = Pattern.compile("<DOCID>\\s*(\\S+)\\s*<");

    /**
     * Regex that finds <"content"> tag
     */
    private static final String CONTENT = "\\s*<[^>]*>\\s*";

    /**
     * Regex that finds <b> tag
     */
    private static final Pattern STRONG = Pattern.compile("<strong(?:\\s+[^>]+)?>\\s*(\\S(?:.*?\\S)?)\\s*</strong>");

    /**
     * Regex that finds hashtags
     */
    private static final Pattern HASHTAGS = Pattern.compile("#\\w+");

    /**
     * Configuration class
     */
    private static final ConfigManager config = ConfigManager.getInstance();



    /**
     * Creates a new Long Eval Corpus document parser.
     *
     * @param in the reader to the document(s) to be parsed.
     * @throws NullPointerException     if {@code in} is {@code null}.
     * @throws IllegalArgumentException if any error occurs while creating the parser.
     */
    public TrecParser(final Reader in) {
        super(new BufferedReader(in));
    }


    @Override
    public boolean hasNext() {

        String id = null;
        String idtmp = null;
        final StringBuilder body = new StringBuilder(BODY_SIZE);
        final StringBuilder start = new StringBuilder(BODY_SIZE);
        final StringBuilder highlights = new StringBuilder(BODY_SIZE);
        final StringBuilder keywords = new StringBuilder(BODY_SIZE);
        try {
            String line;
            boolean cycle = true;
            int cont = -4;
            while (cycle) {
                line = ((BufferedReader) in).readLine();
                if (line != null) {
                    cont++;
                    line = line.concat(" ");
                    if (line.startsWith("</DOC>")) {
                        body.append(line);
                        cycle = false;
                    }
                    Matcher strong = STRONG.matcher(line);
                    while (strong.find()) {
                        highlights.append(strong.group(1)).append(" ");
                    }
                    Matcher hashtags = HASHTAGS.matcher(line);
                    while (hashtags.find()) {
                        highlights.append(hashtags.group(0).replace("#", "")).append(" ");
                    }
                    if (line.contains("<") && !line.contains(">"))
                        line = line.replace("<", "");
                    line = EmojiParser.removeAllEmojis(line);
                    Matcher docno = DOCNO_TAG.matcher(line);
                    Matcher docid = DOC_ID.matcher(line);
                    if (docno.find())
                        id = docno.group(1);
                    else if (docid.find())
                        idtmp = docid.group(1);
                    if (!line.startsWith("<DOCNO>") && !line.startsWith("<DOCID>") && !line.startsWith("<DOC>") && !line.startsWith("<TEXT>") && !line.startsWith("</DOC>") && !line.startsWith("</TEXT>")) {
                        body.append(line);
                        if (cont <= config.getInt("numberOfStartRows") && !line.startsWith("<DOC>") && !line.startsWith("<TEXT>") && line.length() >= 3)
                            start.append(line);
                    }
                } else {
                    next = false;
                    break;
                }
            }

        } catch (IOException e) {
            throw new IllegalStateException("Unable to parse the document.", e);
        }

        if (id != null && !id.equals(idtmp))
            throw new IllegalArgumentException("The two IDs are not equals!");

        if (id != null) {
            // Allow empty document bodies
            String bodyContent = body.toString().replaceAll(CONTENT, "").trim();
            if (bodyContent.isEmpty()) {
                return hasNext();   // Skip empty documents
                //bodyContent = "";  // Treat empty body as an empty string
            }
            document = new ParsedDocument(id, bodyContent, start.toString(), highlights.toString().trim());
        }

        return next;
    }

    @Override
    protected final ParsedDocument parse() {
        return document;
    }


    /**
     * Main method of the class. Just for testing purposes.
     *
     * @param args command line arguments.
     * @throws Exception if something goes wrong while indexing.
     */
    public static void main(String[] args) throws Exception {

        Reader reader = new FileReader(
                "C:\\Users\\lucap\\Documents\\se\\Longeval_2025_Train_Collection_p1\\release_2025_p1\\French\\LongEval Train Collection\\Trec\\2022-06_fr\\part\\collector_kodicare_1.trec");

        TrecParser p = new TrecParser(reader);

        for (ParsedDocument d : p) {
            System.out.printf("%n%n------------------------------------%n%s%n%n%n", d.toString());
        }


    }

}
