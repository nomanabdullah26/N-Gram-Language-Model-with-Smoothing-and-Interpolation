# N-Gram Language Model with Smoothing and Interpolation

## Description

A Python NLP project that builds trigram language models trained on Shakespeare's *Julius Caesar*. It compares Laplace Smoothing and Linear Interpolation for text generation, evaluated using perplexity scores.

## Project Overview

This project implements a statistical N-Gram Language Model from scratch using Python to generate text and evaluate language probabilities. The model is trained on William Shakespeare's *Julius Caesar* from the NLTK Gutenberg corpus.

The project explores two fundamental techniques in Natural Language Processing (NLP):
1. Laplace Smoothing (Add-1 Smoothing) for handling unseen n-grams
2. Linear Interpolation for combining unigram, bigram, and trigram probabilities

These techniques are compared using perplexity, a standard metric for evaluating language models.

## Dataset

- Source: NLTK Gutenberg Corpus
- Text: shakespeare-caesar.txt (William Shakespeare's *Julius Caesar*)
- Vocabulary Size: 3,080 unique words
- Total Tokens: 20,387 words

## Methodology

### 1. Preprocessing
- Converted all text to lowercase
- Removed all non-alphabetic characters using regular expressions
- Tokenized the text into individual words

### 2. N-Gram Model Construction
- Unigram Model: Counts individual word frequencies
- Bigram Model: Counts pairs of consecutive words
- Trigram Model: Counts triples of consecutive words

### 3. Smoothing Techniques

#### Laplace Smoothing (Add-1)
Assigns a small non-zero probability to unseen n-grams using the formula:

P(word | context) = (count + 1) / (total + V)

Where V is the vocabulary size.

#### Linear Interpolation
Combines probabilities from multiple n-gram levels using weighted sum:

P(w | context) = λ3 × P_trigram + λ2 × P_bigram + λ1 × P_unigram

Default weights: λ3 = 0.6, λ2 = 0.3, λ1 = 0.1

### 4. Text Generation
- Greedy Decoding: Selects the word with the highest probability
- Weighted Random Sampling: Selects words based on their probability distribution

### 5. Evaluation
Perplexity is used to measure how well the model predicts a test sentence. Lower perplexity indicates better performance.

Perplexity = exp(-1/N × Σ log P(word | context))

## Results

### Text Generation Comparison

Seed words: "the king"

| Method | Generated Text |
|--------|----------------|
| Laplace Smoothing | the king left spurre cowardice puts conspirator toyles knowst greeuously players perils hearing that vnborne vnluckily sunne mouths lambe rumor mouth marcus satisfied space ore combind tarrying attempt angel sex |
| Linear Interpolation | the king letters ghost early bethinke calculate misconstrued bed bound dint teach must squeale weigh security obserud i winne armes recounts obserue orewatchd howeuer madam mindes glory armes person cowardice |

### Perplexity Comparison

Test Sentence: "the king is dead"

| Model | Perplexity |
|-------|------------|
| Laplace Smoothing | 3080.0000 |
| Linear Interpolation | 956.0138 |

### Key Observation

The interpolation model achieved significantly lower perplexity (956.01) compared to the Laplace smoothing model (3080.00), indicating better predictive performance.

## Analysis and Discussion

### Why Interpolation Performs Better

1. Combines Multiple Levels: Interpolation leverages information from unigrams, bigrams, and trigrams simultaneously, making it more robust when higher-order contexts are sparse.

2. Laplace Smoothing Weakness: Add-1 smoothing assigns uniform probability mass to all unseen words, which drastically weakens predictions when the vocabulary is large.

3. Generalization: The weighted combination in interpolation allows the model to fall back on lower-order n-grams when higher-order contexts are unavailable.

### Limitations

- Greedy Decoding: Selecting the most probable word at each step can lead to repetitive and less diverse text.
- Small Dataset: Training on a single Shakespeare play limits the model's vocabulary and generalization.
- No Semantic Understanding: N-gram models only capture statistical patterns, not meaning.

## Requirements

To run this project, you need Python installed along with the following libraries:

pip install nltk

## How to Run

1. Clone the repository to your local machine.
2. Ensure Python and NLTK are installed.
3. Run the script:

python ngram_model.py

4. The script will:
   - Download the Gutenberg corpus (if not already present)
   - Preprocess the text
   - Build unigram, bigram, and trigram models
   - Generate text using both smoothing methods
   - Compute and compare perplexity scores

## Project Structure

├── ngram_model.py       # Main Python script
├── output.txt           # Sample output log
└── README.md            # Project documentation

## Key Functions

| Function | Description |
|----------|-------------|
| preprocess(text) | Cleans and tokenizes input text |
| build_trigram_model(tokens) | Constructs trigram frequency dictionary |
| laplace_smoothing(trigram, vocab) | Applies Add-1 smoothing to trigram model |
| generate_text(seed, probs, vocab) | Generates text using greedy decoding |
| compute_perplexity(tokens, trigram, vocab) | Calculates perplexity of a test sentence |
| build_unigram_model(tokens) | Constructs unigram frequency dictionary |
| build_bigram_model(tokens) | Constructs bigram frequency dictionary |
| interpolated_probability(...) | Computes linearly interpolated probability |
| generate_text_interpolated(...) | Generates text using weighted sampling |
| compute_perplexity_interpolated(...) | Calculates perplexity using interpolation |

## Conclusion

This project successfully demonstrates the implementation of N-gram language models with two different smoothing strategies. The comparison shows that linear interpolation significantly outperforms Laplace smoothing in terms of perplexity, as it effectively combines information from multiple n-gram levels.

The project provides a solid foundation for understanding core NLP concepts such as:
- Language modeling
- Smoothing techniques
- Text generation
- Perplexity evaluation

## Future Work

- Larger Corpus: Train on a larger and more diverse text corpus for better generalization.
- Kneser-Ney Smoothing: Implement more advanced smoothing techniques like Modified Kneser-Ney.
- Neural Language Models: Compare with RNN-based or Transformer-based language models.
- Beam Search: Replace greedy decoding with beam search for more coherent text generation.
- Hyperparameter Tuning: Optimize interpolation weights (λ1, λ2, λ3) using a validation set.

Note: This project was implemented as part of a Natural Language Processing (NLP) course assignment.
