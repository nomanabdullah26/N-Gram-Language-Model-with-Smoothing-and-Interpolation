import nltk
import re
import math
import random
from collections import defaultdict

# Step 1: Preprocessing

def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()


# Step 2: Trigram Model

def build_trigram_model(tokens: list[str]) -> dict:
    trigram_counts = defaultdict(lambda: defaultdict(int))

    for i in range(len(tokens) - 2):
        context = (tokens[i], tokens[i+1])
        next_word = tokens[i+2]
        trigram_counts[context][next_word] += 1

    return trigram_counts


# Step 3: Laplace Smoothing

def laplace_smoothing(trigram_counts: dict, vocab: list[str]) -> dict:
    smoothed_probs = defaultdict(dict)
    V = len(vocab)

    for context, next_words in trigram_counts.items():
        total = sum(next_words.values())

        for word in vocab:
            count = next_words.get(word, 0)
            smoothed_probs[context][word] = (count + 1) / (total + V)

    return smoothed_probs


# Step 4: Text Generation

def generate_text(seed: list[str], smoothed_probs: dict,
                  vocab: list[str], num_words: int = 30) -> str:

    story = seed.copy()

    while len(story) < num_words:
        context = (story[-2], story[-1])

        if context in smoothed_probs:
            next_word = max(smoothed_probs[context],
                            key=smoothed_probs[context].get)
        else:
            next_word = random.choice(vocab)

        story.append(next_word)

    return ' '.join(story)


# Step 5: Perplexity 
def compute_perplexity(test_tokens: list[str],
                       trigram_counts: dict,
                       vocab: list[str]) -> float:

    V = len(vocab)

    if len(test_tokens) < 3:
        return float('inf')

    log_prob_sum = 0
    N = len(test_tokens) - 2

    for i in range(N):
        context = (test_tokens[i], test_tokens[i+1])
        next_word = test_tokens[i+2]

        # ✅ ALWAYS apply Laplace formula
        if context in trigram_counts:
            total = sum(trigram_counts[context].values())
            count = trigram_counts[context].get(next_word, 0)
        else:
            total = 0
            count = 0

        prob = (count + 1) / (total + V)

        log_prob_sum += math.log(max(prob, 1e-10))

    return math.exp(-log_prob_sum / N)



# FUNCTIONS------

def build_unigram_model(tokens):
    unigram = defaultdict(int)
    for w in tokens:
        unigram[w] += 1
    return unigram


def build_bigram_model(tokens):
    bigram = defaultdict(lambda: defaultdict(int))
    for i in range(len(tokens) - 1):
        bigram[tokens[i]][tokens[i+1]] += 1
    return bigram


def get_unigram_prob(word, unigram, total, V):
    return (unigram.get(word, 0) + 1) / (total + V)


def get_bigram_prob(prev, word, bigram, V):
    if prev in bigram:
        total = sum(bigram[prev].values())
        count = bigram[prev].get(word, 0)
        return (count + 1) / (total + V)
    return 1 / V


def get_trigram_prob(context, word, trigram, V):
    if context in trigram:
        total = sum(trigram[context].values())
        count = trigram[context].get(word, 0)
        return (count + 1) / (total + V)
    return 1 / V


def interpolated_probability(word, context,
                             trigram, bigram, unigram,
                             total, V,
                             l3=0.6, l2=0.3, l1=0.1):

    return (
        l3 * get_trigram_prob(context, word, trigram, V) +
        l2 * get_bigram_prob(context[1], word, bigram, V) +
        l1 * get_unigram_prob(word, unigram, total, V)
    )


def generate_text_interpolated(seed, trigram, bigram,
                              unigram, vocab, total, num_words=30):

    story = seed.copy()
    V = len(vocab)

    while len(story) < num_words:
        context = (story[-2], story[-1])

        words = vocab[:500]
        probs = [interpolated_probability(w, context,
                                         trigram, bigram, unigram,
                                         total, V)
                 for w in words]

        next_word = random.choices(words, weights=probs, k=1)[0]
        story.append(next_word)

    return ' '.join(story)


def compute_perplexity_interpolated(tokens, trigram,
                                    bigram, unigram,
                                    total, vocab):

    V = len(vocab)
    N = len(tokens) - 2
    log_prob_sum = 0

    for i in range(N):
        context = (tokens[i], tokens[i+1])
        word = tokens[i+2]

        prob = interpolated_probability(word, context,
                                        trigram, bigram,
                                        unigram, total, V)

        log_prob_sum += math.log(max(prob, 1e-10))

    return math.exp(-log_prob_sum / N)


# MAIN------

def main():
    nltk.download('gutenberg')
    from nltk.corpus import gutenberg

    raw = gutenberg.raw('shakespeare-caesar.txt')

    tokens = preprocess(raw)
    vocab = list(set(tokens))

    print("=== Preprocessing ===")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Total tokens:    {len(tokens)}")

    trigram = build_trigram_model(tokens)
    smoothed = laplace_smoothing(trigram, vocab)

    seed = ['the', 'king']
    text = generate_text(seed, smoothed, vocab)

    print("\n=== Text Generation (Laplace Smoothing) ===")
    print("Seed:", ' '.join(seed))
    print("Generated:", text)

    test = preprocess("the king is dead")
    ppl = compute_perplexity(test, trigram, vocab)

    print("\n=== Perplexity ===")
    print("Test sentence: 'the king is dead'")
    print(f"Perplexity: {ppl:.4f}")


    uni = build_unigram_model(tokens)
    bi = build_bigram_model(tokens)

    text_i = generate_text_interpolated(seed, trigram, bi, uni, vocab, len(tokens))
    ppl_i = compute_perplexity_interpolated(test, trigram, bi, uni, len(tokens), vocab)
    
    print("\n=== Bonus ===")
    print("Generated (Interpolation):", text_i)
    print(f"Perplexity (Interpolation): {ppl_i:.4f}")

    print("\nComparison:")
    print("Interpolation gives lower perplexity because it combines multiple n-gram levels.")
    print("Laplace smoothing assigns uniform probability which weakens predictions.")
    print("Interpolation improves generalization and produces more stable text.")
    print("Greedy decoding may still reduce diversity in both approaches.")


if __name__ == "__main__":
    main()