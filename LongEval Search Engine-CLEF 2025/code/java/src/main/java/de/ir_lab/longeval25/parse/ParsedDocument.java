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

package de.ir_lab.longeval25.parse;

import org.apache.commons.lang3.builder.ToStringBuilder;
import org.apache.commons.lang3.builder.ToStringStyle;
import org.apache.lucene.document.Field;

/**
 * Represents a parsed document to be indexed.
 *
 * @author Searchill
 * @version 1.01
 * @since 1.00
 */
public class ParsedDocument {


    /**
     * The names of the {@link Field}s within the index.
     *
     * @author Searchill
     * @version 1.01
     * @since 1.00
     */
    public static final class FIELDS {

        private FIELDS () {}

        /**
         * The document identifier
         */
        public static final String ID = "id";

        /**
         * The document identifier
         */
        public static final String BODY = "body";

        /**
         * The document first lines
         */
        public static final String START = "start";

        /**
         * The document highlights words
         */
        public static final String HIGHLIGHTS = "highlights";
    }


    /**
     * The unique document identifier.
     */
    private final String id;

    /**
     * The body of the document.
     */
    private final String body;

    /**
     * The start of the document
     */
    private final String start;

    /**
     * The document highlights words
     */
    public final String highlights;

    /**
     * Creates a new parsed document
     *
     * @param id   the unique document identifier.
     * @param body the body of the document.
     * @param start the first lines of the document.
     * @param highlights the highlighted words of the document.
     * @throws NullPointerException  if {@code id} and/or {@code body} are {@code null}.
     * @throws IllegalStateException if {@code id} and/or {@code body} are empty.
     */
    public ParsedDocument(final String id, final String body, final String start, final String highlights) {

        if (id == null) {
            throw new NullPointerException("Document identifier cannot be null.");
        }

        if (id.isEmpty()) {
            throw new IllegalStateException("Document identifier cannot be empty.");
        }

        this.id = id;

        if (body == null) {
            throw new NullPointerException("Document body cannot be null.");
        }

        if (body.isEmpty()) {
            throw new IllegalStateException("Document body cannot be empty.");
        }

        this.body = body;

        if (start == null) {
            throw new NullPointerException("Document body cannot be null.");
        }

        this.start = start;

        if (highlights == null) {
            throw new NullPointerException("Document body cannot be null.");
        }

        this.highlights = highlights;
    }

    /**
     * Returns the unique document identifier.
     *
     * @return the unique document identifier.
     */
    public String getIdentifier() {
        return id;
    }

    /**
     * Returns the body of the document.
     *
     * @return the body of the document.
     */
    public String getBody() {
        return body;
    }

    /**
     * Returns the start of the document.
     *
     * @return the start of the document.
     */
    public String getStart() {
        return start;
    }

    /**
     * Returns the highlights of the document.
     *
     * @return the highlights of the document.
     */
    public String getHighlights() {
        return highlights;
    }

    @Override
    public final String toString() {
        ToStringBuilder tsb = new ToStringBuilder(this, ToStringStyle.MULTI_LINE_STYLE).append("identifier", id).append(
                "body", body).append("start", start).append("highlights", highlights);

        return tsb.toString();
    }

    @Override
    public final boolean equals(Object o) {
        return (this == o) || ((o instanceof ParsedDocument) && id.equals(((ParsedDocument) o).id));
    }

    @Override
    public final int hashCode() {
        return 37 * id.hashCode();
    }


}
