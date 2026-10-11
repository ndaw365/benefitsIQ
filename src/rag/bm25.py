"""Okapi BM25 keyword ranking over chunk texts, written by hand (no search library)."""

import math
import re
from collections import Counter

# Question words that carry no topic. IDF already down-weights common words; this just removes noise.
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it its me my of on or "
    "the their this to under what when which who will with".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens minus stopwords. 'self-inflicted' -> ['self', 'inflicted']."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    """Score documents for a query: sum over query terms of IDF x saturated, length-normalized term frequency."""

    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.ids = ids
        self.k1 = k1  # how fast repeated terms stop adding score (saturation)
        self.b = b  # how strongly long chunks are penalized (0 = not at all, 1 = fully)
        self.term_freqs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_length = sum(self.lengths) / len(self.lengths) if texts else 0.0
        n = len(texts)
        doc_freq = Counter(term for tf in self.term_freqs for term in tf)
        # "+1" inside the log keeps IDF positive even for terms in more than half the chunks.
        self.idf = {term: math.log((n - df + 0.5) / (df + 0.5) + 1) for term, df in doc_freq.items()}

    def search(self, query: str, top_n: int = 20) -> list[tuple[str, float]]:
        """Return up to top_n (id, score) pairs with score > 0, best first."""
        terms = tokenize(query)
        scored = []
        for doc_id, tf, length in zip(self.ids, self.term_freqs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            score = sum(
                self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf
            )
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n]
