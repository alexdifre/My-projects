# LongEval CLEF 2025 Lab

**LongEval-WebRetrieval Project of group SEARCHILL for the "Search Engines" course, A.Y. 2024/2025.**

An advanced information retrieval system for the LongEval-Retrieval Task at CLEF 2025, combining classical IR techniques with modern neural re-ranking approaches.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Java](https://img.shields.io/badge/Java-11+-orange.svg)](https://www.oracle.com/java/)
[![Lucene](https://img.shields.io/badge/Apache%20Lucene-9.0+-green.svg)](https://lucene.apache.org/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-yellow.svg)](LICENSE)

## Organisation of the repository

The repository is organised as follows:

- **code**: this folder contains the source code of the developed system.
- **runs**: this folder contains the runs produced by the developed system:
    - in the **LongEval runs** folder, you can find the runs submitted to CLEF.
    - in the **train runs** folder, you can find the train runs results.
- **homework-1**: this folder contains the report describing the techniques applied.
**homework-2**: this folder contains the report describing the results 
- **LongEval Search Engine-CLEF 2025**: this is the final paper submitted to CLEF
A comprehensive explanation of the technical intricacies can be found in the associated paper: **LongEval Search Engine-CLEF 2025**
---

## Overview

This project implements a robust and efficient search engine designed to handle the dynamic nature of web search, developed for the CLEF 2025 LongEval Lab. The system leverages the comprehensive LongEval Websearch collection and combines traditional information retrieval methods with state-of-the-art neural re-ranking techniques.

Our approach focuses on optimizing both query and document processing pipelines to maximize ranking efficacy and deliver highly relevant search results. The system demonstrates strong performance with MAP scores up to 0.2585 and nDCG scores up to 0.3601 on the training dataset.

---

---

## Table of Contents

1. [Key Features](#key-features)
2. [System Architecture](#system-architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Usage](#usage)
5. [Performance Results](#performance-results)
6. [Technical Implementation](#technical-implementation)
7. [Experimental Analysis](#experimental-analysis)
8. [Future Work](#future-work)

---

## Key Features

### Core Capabilities
- Advanced French text analysis with custom tokenization and filtering
- Multi-strategy query expansion (dictionary, synonyms, fuzzy, phrase)
- Neural re-ranking using Cohere multilingual-v3
- Heuristic-based document quality assessment
- Support for TREC-formatted document collections
- Efficient indexing with field-specific boosting

### Technical Highlights
- Projection-free optimization techniques
- BM25 similarity with neural re-ranking combination
- Part-of-Speech (POS) based compound word detection
- Abbreviation expansion for query normalization
- Word2Vec and LLM-based semantic expansion
- Pseudo-Relevance Feedback (PRF) implementation

### Performance Metrics
- **Best MAP:** 0.2585 (training set 2023-01)
- **Best nDCG:** 0.3601 (training set 2023-01)
- **Index Size:** 7.5 GB (French 2022-06), 12.4 GB (French 2023-01)
- **Processing Speed:** Efficient on consumer hardware (AMD Ryzen 5500U)

---

## System Architecture

### 1. Parser Component

Responsible for extracting structured data from TREC-formatted documents.

**Key Classes:**
- **DocumentParser:** Abstract base class for document iteration
- **ParsedDocument:** Encapsulates document structure (ID, body, start, highlights)
- **TrecParser:** Specialized TREC format handler

**Features:**
- Regex-based field extraction (`<DOCNO>`, `<DOCID>`)
- Multi-stage buffering for large document bodies
- Hashtag and strong-tag highlight extraction
- Emoji and URL filtering
- Robust error handling for malformed data

```java
public class TrecParser extends DocumentParser {
    @Override
    public ParsedDocument parse(InputStream stream) {
        // Extract document fields using regex patterns
        // Buffer document content efficiently
        // Extract emphasized content and highlights
        return new ParsedDocument(id, body, start, highlights);
    }
}
```

### 2. Analyzer Component

Custom French text analyzer with sophisticated token processing pipeline.

**Pipeline Sequence:**
1. **Tokenization** (WhiteSpace/Letter/Standard/OpenNLP)
2. **LowerCaseFilter** - Case normalization
3. **RepeatedLetterFilter** - Handle letter repetitions
4. **AbbreviationExpansionFilter** - Expand known abbreviations
5. **ICUFoldingFilter** - Unicode harmonization
6. **NBSPFilter** - Whitespace normalization
7. **ElisionFilter** - French contraction handling
8. **RemoveDuplicatesTokenFilter** - Remove duplicate tokens
9. **CompoundPOSTokenFilter** - POS-based compound detection
10. **ShingleFilter** - N-gram generation
11. **LengthFilter** - Token length constraints
12. **StopFilter** - Stopword removal (40 French + 10 English terms)
13. **PositionFilter** - Position increment adjustment

**Stemming Options:**
- **SnowballFilter** - Multi-language Snowball algorithm
- **FrenchLightStemFilter** - Light stemming for French
- **FrenchMinimalStemFilter** - Minimal morphological reduction

### 3. Indexer Component

Multi-field indexing with strategic storage decisions.

**Indexed Fields:**

| Field | Tokenized | Frequency | Positions | Stored | Boost |
|-------|-----------|-----------|-----------|--------|-------|
| Body | ✓ | ✓ | ✓ | ✓ | 1.0× |
| Start | ✓ | ✓ | ✗ | ✓ | 1.5× |
| Highlights | ✓ | ✓ | ✗ | ✓ | 1.2× |

**Indexing Statistics:**

| Collection | Size (GB) | Documents | Body Terms | Index Size (GB) |
|------------|-----------|-----------|------------|-----------------|
| French 2022-06 | 9.3 | 1,590,024 | 3,423,470 | 7.5 |
| French 2023-01 | 12.8 | 2,537,554 | 38,313,777 | 12.4 |

### 4. Searcher Component

Advanced query processing with multiple expansion strategies.

**Query Construction:**
```java
BooleanQuery.Builder builder = new BooleanQuery.Builder();

// Mandatory base query
builder.add(originalQuery, BooleanClause.Occur.MUST);

// Field boosting
builder.add(boost(startQuery, 1.5f), BooleanClause.Occur.SHOULD);
builder.add(boost(highlightQuery, 1.2f), BooleanClause.Occur.SHOULD);

// Query expansion (optional)
builder.add(synonymQuery, BooleanClause.Occur.SHOULD);  // 0.1× weight
builder.add(fuzzyQuery, BooleanClause.Occur.SHOULD);    // 0.2-0.8× weight
builder.add(phraseQuery, BooleanClause.Occur.SHOULD);   // slop=5
```

**Expansion Strategies:**

1. **Word N-grams**
   - Generates bigrams from query terms
   - Example: "running shoes women" → ["running shoes", "shoes women"]

2. **Fuzzy Search**
   - Edit distance = 2
   - Handles typos and misspellings
   - Example: "Adibas" matches "Adidas"

3. **Dictionary-based Splitting**
   - Handles concatenated terms
   - Uses synonym dictionary for sub-term extraction
   - Example: "term1.term2.term3" → ["term1", "term2", "term3"]

4. **Synonym Expansion**
   - Random selection of 3 synonyms
   - Weighted by field importance
   - Start field: 1.5×, Highlights: 1.2×

5. **Word2Vec Semantic Expansion**
   - 300-dimensional French fastText embeddings
   - Trained on Common Crawl corpus
   - Captures semantic relationships

6. **LLM-based Expansion**
   - ChatGPT-3.5-turbo / DeepSeek APIs
   - Generates 5-10 related terms
   - 0.4× weight for generated terms

7. **Pseudo-Relevance Feedback (PRF)**
   - Expands query using top-ranked documents
   - Extracts relevant terms from initial results

8. **Phrase Queries**
   - Exact sequence matching
   - Slop window = 5 positions
   - Preserves term order

### 5. Re-ranking Pipeline

Neural re-ranking for final result optimization.

**Process:**
1. Normalize BM25 scores to [0,1]
2. Retrieve top 50 document bodies
3. Apply Cohere multilingual-v3 re-ranker
4. Combine scores: `finalScore = 0.4 × BM25 + 0.6 × Reranker`

**Heuristic Re-ranker:**
- **Length Penalty:** Penalize very short or excessively long documents
- **Vocabulary Penalty:** Assess lexical richness (unique terms / total terms)
- **Spam Detection:** Penalize keyword stuffing
- **Repetition Penalty:** Identify low-quality content
- **Query-specific Heuristics:** Boost longer documents for short queries

---

## Installation and Setup

### Prerequisites

**Software Requirements:**
- Java 11+
- Python 3.8+
- Apache Lucene 9.0+
- Maven 3.6+

**Python Dependencies:**
- Flask (HTTP server)
- Gensim (Word2Vec)
- OpenAI API / DeepSeek API (optional, for LLM expansion)
- Cohere API (optional, for neural re-ranking)

### Installation Steps

```bash
# Clone the repository
git clone https://bitbucket.org/upd-dei-stud-prj/seupd2425-searchill/src/master/
cd seupd2425-searchill

# Build Java components
mvn clean install

# Install Python dependencies
pip install -r requirements.txt

# Download Word2Vec embeddings
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz
gunzip cc.fr.300.vec.gz
```

### Configuration

**LanguageTool Setup (Query Correction):**
```bash
# Option 1: Use public API
export LANGUAGETOOL_URL="https://api.languagetoolplus.com/v2"

# Option 2: Run local instance
docker run -p 8010:8010 erikvl87/languagetool
export LANGUAGETOOL_URL="http://localhost:8010/v2"
```

**Word2Vec Server:**
```bash
# Start the Flask server
cd python_services
python word2vec_server.py --model cc.fr.300.vec --port 5000
```

**API Keys (Optional):**
```bash
export OPENAI_API_KEY="your_key_here"
export COHERE_API_KEY="your_key_here"
```

---

## Usage

### Basic Search

```bash
# Run the search engine
java -jar target/searchengine.jar \
    --index data/index_2023-01 \
    --query "chaussures de course"
```

### Batch Evaluation

```bash
# Evaluate on TREC topics
java -jar target/searchengine.jar \
    --index data/index_2023-01 \
    --topics data/topics_2023-01.xml \
    --output results/run_baseline.txt
```

### Custom Configuration

```java
// Configure analyzer
Analyzer analyzer = new FrenchAnalyzer(
    TokenizerType.LETTER,
    StemmerType.SNOWBALL,
    stopwordList
);

// Configure searcher
SearcherConfig config = SearcherConfig.builder()
    .enableFuzzy(true)
    .enablePhrase(true)
    .enableSynonyms(true)
    .enableReranker(true)
    .rerankerTopK(50)
    .build();

// Run search
Searcher searcher = new Searcher(indexDir, analyzer, config);
SearchResults results = searcher.search(query, maxResults);
```

### API Endpoints

**Word2Vec Similarity:**
```bash
curl http://localhost:5000/similar?word=chaussures&topn=5
```

**LLM Query Expansion:**
```bash
curl -X POST http://localhost:5001/expand \
    -H "Content-Type: application/json" \
    -d '{"query": "running shoes", "num_terms": 5}'
```

---

## Performance Results

### Training Results (2023-01 Dataset)

| Configuration | MAP | nDCG |
|--------------|------|------|
| Baseline (Letter Tokenizer + StopList) | 0.2002 | 0.3061 |
| + Phrase Query | 0.2243 | 0.3270 |
| + Fuzzy Search | 0.2135 | 0.3230 |
| + Start Field Boost | 0.2011 | 0.3069 |
| Phrase + Fuzzy + Start | 0.2359 | 0.3422 |
| Phrase + Fuzzy + Start + Highlights | 0.2359 | 0.3423 |
| **Best: + Neural Reranker** | **0.2585** | **0.3601** |

### Component Analysis

**Tokenizer Performance:**

| Tokenizer | MAP | nDCG | Notes |
|-----------|-----|------|-------|
| WhiteSpace | 0.1855 | 0.2847 | Poor performance |
| **Letter** | **0.2002** | **0.3061** | Best choice |
| Standard | 0.2002 | 0.3059 | Similar to Letter |
| OpenNLP | 0.1892 | 0.2906 | Slow, less effective |

**Stemmer Performance:**

| Stemmer | MAP | nDCG | Recommendation |
|---------|-----|------|----------------|
| FrenchLight | 0.1722 | 0.2745 | Not recommended |
| FrenchMinimal | 0.1798 | 0.2832 | Not recommended |
| Snowball | 0.1848 | 0.2903 | Not recommended |

*Note: All stemmers decreased performance and were excluded from final configuration.*

**Feature Impact:**

| Feature | MAP | nDCG | Status |
|---------|-----|------|--------|
| Dictionary | 0.2014 | 0.3072 | ✓ Beneficial alone, ✗ in combination |
| Synonyms | 0.1979 | 0.3034 | Neutral |
| Fuzzy | 0.2135 | 0.3230 | ✓ Strong improvement |
| Phrase | 0.2243 | 0.3270 | ✓ Strong improvement |
| PRF | 0.1768 | 0.2905 | ✗ Decreased performance |
| Word2Vec | 0.1689 | 0.2814 | ✗ Poor results |
| N-grams | 0.2002 | 0.3061 | Neutral |
| Heuristic | 0.1361 | 0.2237 | ✗ Worst performance |
| Neural Reranker | 0.2585 | 0.3601 | ✓ Best improvement |

### Test Results (Held-out Collections)

| Month | Baseline (MAP/nDCG) | + Reranker (MAP/nDCG) | + Correction + Reranker (MAP/nDCG) |
|-------|---------------------|----------------------|-----------------------------------|
| 2023/03 | 0.2394 / 0.3449 | 0.2706 / 0.3703 | 0.2707 / 0.3704 |
| 2023/04 | 0.2566 / 0.3682 | 0.2804 / 0.3865 | 0.2800 / 0.3861 |
| 2023/05 | 0.2515 / 0.3603 | 0.2802 / 0.3830 | 0.2796 / 0.3823 |
| 2023/06 | 0.2685 / 0.3756 | 0.2973 / 0.3985 | 0.2970 / 0.3982 |
| 2023/07 | 0.2475 / 0.3539 | 0.2715 / 0.3736 | 0.2706 / 0.3725 |
| 2023/08 | 0.2208 / 0.3095 | 0.2375 / 0.3209 | 0.2355 / 0.3207 |

**Key Observations:**
- Neural re-ranking provides consistent improvements across all months
- Query correction adds minimal but consistent gains
- August 2023 shows anomalously lower performance (unexplained)
- ANOVA tests confirm statistically significant system differences (p < 0.0001)

---

## Experimental Analysis

### Statistical Significance (ANOVA)

**Results for all test months show:**
- **System effect:** Highly significant (p < 1e-16) for both nDCG and MAP
- **Topic effect:** Significant, confirming query-dependent variance
- **Error variance:** Low relative to system effects, indicating reliable results

**Example (2023/06):**
```
nDCG Analysis:
- System SS: 2.4794, F: 100.90, p < 1e-16
- Topic SS: 2233.72, F: 25.24, p < 1e-16
- Error SS: 177.01

AP Analysis:
- System SS: 3.9272, F: 110.63, p < 1e-16
- Topic SS: 2471.65, F: 19.33, p < 1e-16
- Error SS: 255.71
```

### Component Contributions

**Primary Performance Drivers:**
1. **Neural Re-ranker** - Largest single improvement (+9.6% MAP, +5.2% nDCG)
2. **Phrase Matching** - Strong baseline improvement (+12.0% MAP, +6.8% nDCG)
3. **Fuzzy Search** - Handles typos and variations (+6.6% MAP, +5.5% nDCG)
4. **Field Boosting** - Start and highlights provide modest gains (+0.4% MAP)

**Ineffective Components:**
- Word2Vec expansion (-15.7% MAP)
- PRF (-11.7% MAP)
- Heuristic re-ranker (-32.0% MAP)
- All stemming approaches (-5% to -13% MAP)

### Failure Analysis

**Word2Vec Limitations:**
- Pre-trained embeddings may not match domain
- Semantic drift introduces noise
- No context-aware selection of similar terms

**PRF Issues:**
- Initial retrieval quality affects expansion
- May amplify query drift
- Requires careful parameter tuning

**Heuristic Re-ranker Problems:**
- Multiple penalty interactions cancel benefits
- Difficult to find optimal parameter combinations
- Domain-specific heuristics don't generalize

**POS Compound Detection:**
- Limited grammar rule coverage
- Missing NER model (Apache OpenNLP discontinued support)
- Could not capture all multi-word expressions

---

## Future Work

### Immediate Improvements

1. **Advanced Re-ranking**
   - Experiment with BERT-based models
   - Cross-encoder architectures
   - Ensemble re-ranking strategies
   - Larger re-ranking windows (top-100, top-200)

2. **LLM-based Expansion**
   - Use more advanced models (GPT-4, Claude)
   - Implement prompt engineering for better term generation
   - Context-aware expansion based on initial results
   - Cost-effective alternatives (Mistral, Llama)

3. **Named Entity Recognition**
   - Integrate modern NER models (spaCy, Flair)
   - Extract and index entities separately
   - Entity-aware query expansion
   - Cross-document entity resolution

4. **Document Keyword Extraction**
   - TF-IDF based keyword extraction
   - TextRank / RAKE algorithms
   - Neural keyword generation
   - Query-document keyword matching

---

## Hardware Specifications

**Test Environment:**
- **Model:** Asus Vivobook X515
- **CPU:** AMD Ryzen 5500U (6 cores)
- **GPU:** Radeon Graphics
- **RAM:** 12 GB LPDDR5 @ 3200 MHz

---

## Data and Resources

**Datasets:**
- LongEval CLEF 2025 Lab: [https://clef-longeval.github.io/data/](https://clef-longeval.github.io/data/)

**Source Code:**
- Bitbucket Repository: [seupd2425-searchill](https://bitbucket.org/upd-dei-stud-prj/seupd2425-searchill/src/master/)

**External Services:**
- LanguageTool: [https://api.languagetoolplus.com/v2](https://api.languagetoolplus.com/v2)
- LanguageTool Local: [https://dev.languagetool.org/http-server](https://dev.languagetool.org/http-server)
- Cohere: [https://cohere.com/](https://cohere.com/)
- FastText Embeddings: [Facebook AI Research](https://fasttext.cc/)

---

## Technical Stack

**Core Technologies:**
- **Java 11+** - Main search engine implementation
- **Apache Lucene 9.0+** - Indexing and retrieval
- **Python 3.8+** - Neural components and services
- **Flask** - Web service framework
- **Gensim** - Word embedding utilities
- **OpenNLP** - NLP processing
- **Maven** - Build and dependency management

**APIs and Services:**
- Cohere API (neural re-ranking)
- OpenAI API (query expansion)
- LanguageTool (query correction)

---
### Authors

- **Alessandro Di Frenna** (alessandro.difrenna@studenti.unipd.it)
- **Luca Pellegrini** (luca.pellegrini@studenti.unipd.it)
- **Nicola Ferro** (nicola.ferro@unipd.it) - Supervisor

**Institution:** University of Padua, Department of Maths and Information Engineering, Italy

## Group Contributions

**Alessandro Di Frenna:**
- Tokenizer testing and evaluation
- Implementation of AbbreviationExpansionFilter
- Development of CompoundPOSTokenFilter using Apache OpenNLP
- HeuristicReranker implementation
- Performance analysis and optimization

**Luca Pellegrini:**
- Parser development (TREC format handling)
- Start and highlight field features
- Stoplist construction and optimization
- Searcher component tuning and evaluation
- Experimental runs and result analysis

**Nicola Ferro:**
- Project supervision and guidance
- Research direction and methodology
- Academic oversight

---

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{difrenna2025seupd,
  title={SEUPD@CLEF: Team Searchill Notebook for the LongEval Lab at CLEF 2025},
  author={Di Frenna, Alessandro and Pellegrini, Luca and Ferro, Nicola},
  booktitle={CLEF 2025: Conference and Labs of the Evaluation Forum},
  year={2025},
  address={Madrid, Spain},
  month={September}
}
```

---

## License

This work is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

---

## Acknowledgments

We thank the CLEF 2025 LongEval Lab organizers for providing the datasets and evaluation framework. Special thanks to the University of Padua Department of Data Science and the SearchEngines course for the foundational knowledge and resources that made this project possible.

---

**University of Padua | Department of Maths and Information Engineering | CLEF 2025**

