import numpy as np
import re
from typing import List, Tuple
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# make sure in the notebook you called:
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))

def _tokens(text: str) -> List[str]:
    return [t.lower() for t in word_tokenize(text)]

def _char_ngrams(text: str, n: int = 3) -> List[str]:
    text = text.lower()
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def _overlap_coef(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def lexical_features(s1: str, s2: str) -> np.ndarray:
    t1 = _tokens(s1)
    t2 = _tokens(s2)

    set1, set2 = set(t1), set(t2)
    len1, len2 = len(t1), len(t2)

    # n-grams
    b1 = set(zip(t1, t1[1:]))
    b2 = set(zip(t2, t2[1:]))

    c3_1 = set(_char_ngrams(s1, 3))
    c3_2 = set(_char_ngrams(s2, 3))

    # stopword filtering
    c1 = {w for w in t1 if w not in STOPWORDS}
    c2 = {w for w in t2 if w not in STOPWORDS}

    # numbers
    nums1 = set(re.findall(r"\d+(\.\d+)?", s1))
    nums2 = set(re.findall(r"\d+(\.\d+)?", s2))

    feats = []

    # 1–2: unigram overlap
    feats.append(_jaccard(set1, set2))
    feats.append(_overlap_coef(set1, set2))

    # 3: bigram jaccard
    feats.append(_jaccard(b1, b2))

    # 4: char 3-gram jaccard
    feats.append(_jaccard(c3_1, c3_2))

    # 5–6: length-based
    if max(len1, len2) > 0:
        feats.append(min(len1, len2) / max(len1, len2))
        feats.append(abs(len1 - len2) / max(len1, len2))
    else:
        feats.extend([1.0, 0.0])

    # 7: content word Jaccard
    feats.append(_jaccard(c1, c2))

    # 8: numeric overlap + mismatch indicator
    feats.append(_jaccard(nums1, nums2))
    feats.append(0.0 if nums1 == nums2 else 1.0)

    return np.array(feats, dtype=float)


def syntactic_features(s1: str, s2: str) -> np.ndarray:
    t1 = _tokens(s1)
    t2 = _tokens(s2)

    pos1 = [p for _, p in nltk.pos_tag(t1)] if t1 else []
    pos2 = [p for _, p in nltk.pos_tag(t2)] if t2 else []

    set_pos1, set_pos2 = set(pos1), set(pos2)

    # bigrams of POS
    bpos1 = set(zip(pos1, pos1[1:]))
    bpos2 = set(zip(pos2, pos2[1:]))

    def prop(tags, prefixes):
        if not tags:
            return 0.0
        count = sum(1 for t in tags if any(t.startswith(p) for p in prefixes))
        return count / len(tags)

    noun1 = prop(pos1, ["NN"])
    noun2 = prop(pos2, ["NN"])

    verb1 = prop(pos1, ["VB"])
    verb2 = prop(pos2, ["VB"])

    adjadv1 = prop(pos1, ["JJ", "RB"])
    adjadv2 = prop(pos2, ["JJ", "RB"])

    # coarse pattern: map tags to N/V/A/O and compress
    def coarse_pattern(tags):
        def map_tag(t):
            if t.startswith("NN"):
                return "N"
            if t.startswith("VB"):
                return "V"
            if t.startswith("JJ"):
                return "A"
            if t.startswith("RB"):
                return "R"
            return "O"
        seq = [map_tag(t) for t in tags]
        # compress runs: N N V -> N V
        comp = []
        for x in seq:
            if not comp or comp[-1] != x:
                comp.append(x)
        return set(comp)

    coarse1 = coarse_pattern(pos1)
    coarse2 = coarse_pattern(pos2)

    feats = []
    # 1: POS unigram Jaccard
    feats.append(_jaccard(set_pos1, set_pos2))
    # 2: POS bigram Jaccard
    feats.append(_jaccard(bpos1, bpos2))
    # 3–5: proportion diffs
    feats.append(abs(noun1 - noun2))
    feats.append(abs(verb1 - verb2))
    feats.append(abs(adjadv1 - adjadv2))
    # 6: syntactic length ratio
    if max(len(pos1), len(pos2)) > 0:
        feats.append(min(len(pos1), len(pos2)) / max(len(pos1), len(pos2)))
    else:
        feats.append(1.0)
    # 7: coarse POS pattern overlap
    feats.append(_jaccard(coarse1, coarse2))

    return np.array(feats, dtype=float)


def combined_features(s1: str, s2: str) -> np.ndarray:
    return np.concatenate([lexical_features(s1, s2),
                           syntactic_features(s1, s2)])
