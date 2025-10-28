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

package de.ir_lab.longeval25.index;

import de.ir_lab.longeval25.parse.ParsedDocument;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.IndexOptions;

import java.io.Reader;

/**
 * Represents a {@link Field} for containing the body of a document.
 * <p>
 * It is a tokenized field, not stored, keeping only document ids and term frequencies (see {@link
 * IndexOptions#DOCS_AND_FREQS} in order to minimize the space occupation.
 *
 * @version 1.00
 * @since 1.00
 */
public class BodyField extends Field {

    /**
     * The type of the document body field
     */
    private static final FieldType BODY_TYPE = new FieldType();

    static {
        BODY_TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS);
        BODY_TYPE.setTokenized(true);
        BODY_TYPE.setStored(true);
    }


    /**
     * Create a new field for the body of a document.
     *
     * @param value the contents of the body of a document.
     */
    public BodyField(final Reader value) {
        super(ParsedDocument.FIELDS.BODY, value, BODY_TYPE);
    }

    /**
     * Create a new field for the body of a document.
     *
     * @param value the contents of the body of a document.
     */
    public BodyField(final String value) {
        super(ParsedDocument.FIELDS.BODY, value, BODY_TYPE);
    }

}

/**
 * StartField represents a field for the start/title content of a document.
 * It is designed to be indexed with document frequency and term frequency.
 */
class StartField extends Field {

    /**
     * The type of document field to be indexed
     */
    private static final FieldType START_TYPE = new FieldType();

    static {
        START_TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
        START_TYPE.setTokenized(true);
        START_TYPE.setStored(true);
    }

    /**
     * Creates a new field for the title/start
     *
     * @param value the title
     */
    public StartField(final String value) {
        super(ParsedDocument.FIELDS.START, value, START_TYPE);
    }
}

/**
 * StartField represents a field for the highlight content of a document.
 * It is designed to be indexed with document frequency and term frequency.
 */
class HighlightedField extends Field {

    /**
     * The type of document field to be indexed
     */
    private static final FieldType HIGHLIGHTS_TYPE = new FieldType();

    static {
        HIGHLIGHTS_TYPE.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
        HIGHLIGHTS_TYPE.setTokenized(true);
        HIGHLIGHTS_TYPE.setStored(true);
    }

    /**
     * Creates a new field for the highlighted words
     *
     * @param value the title
     */
    public HighlightedField(final String value) {
        super(ParsedDocument.FIELDS.HIGHLIGHTS, value, HIGHLIGHTS_TYPE);
    }
}
