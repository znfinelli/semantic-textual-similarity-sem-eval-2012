"""
features.py
-----------
Feature Engineering module for STS (Semantic Textual Similarity).

SCIENTIFIC JUSTIFICATION:
This module implements a "Supervised Feature Engineering" approach inspired by the 
top performing systems at SemEval-2012 Task 6:
1. UKP Lab (Rank 1): Emphasized structural string similarity (LCS) and N-grams.
2. TakeLab (Rank 2): Emphasized syntactic dependencies, number matching, and WordNet path similarity.

The pipeline adheres to the "Resource-Light" constraint by avoiding large embedding models (BERT/LSA) 
and relying on explicit linguistic knowledge (WordNet, Dependency Parsing).
"""

import numpy as np
import re
import nltk
import spacy
import spacy.cli
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from difflib import SequenceMatcher

# ==========================================
# 1. AUTO-SETUP
# ==========================================
def setup_resources():
    """
    Automatically checks and downloads necessary NLTK/spaCy resources.
    Ensures reproducibility across different environments without manual setup.
    """
    print("--- Checking and Downloading Resources ---")
    resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 
                 'wordnet', 'omw-1.4', 'stopwords']
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except:
            # Fallback if quiet download fails
            nltk.download(res)

    model_name = "en_core_web_sm"
    if not spacy.util.is_package(model_name):
        print(f"Downloading spaCy model '{model_name}'...")
        spacy.cli.download(model_name)

# Run setup on import
setup_resources()

# ==========================================
# 2. GLOBAL RESOURCES
# ==========================================
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print(f"Warning: spaCy model not found ({e}). Syntactic features will return 0.")
    nlp = None

# ==========================================
# 3. PREPROCESSING
# ==========================================

def normalize_time(text: str) -> str:
    """Normalizes various time formats to 00:00 (Standardization)"""
    flags = re.IGNORECASE
    text = re.sub(r'(\d{1,2})[hH:.](\d{2})', r'\1:\2', text, flags=flags)
    text = re.sub(r'\b(\d):(\d{2})\b', r'0\1:\2', text, flags=flags)
    text = re.sub(r'(\d{1,2})h\b', r'\1:00', text, flags=flags)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _penntag_to_wordnet(tag):
    """Maps Penn Treebank tags to WordNet tags for accurate lemmatization."""
    if tag.startswith('N'): return wn.NOUN
    if tag.startswith('V'): return wn.VERB
    if tag.startswith('J'): return wn.ADJ
    if tag.startswith('R'): return wn.ADV
    return None

def preprocess(text: str) -> list:
    """
    Standard NLP Pipeline:
    Time Norm -> Tokenize -> Lower -> Remove Stop/Punct -> Lemmatize
    """
    text = normalize_time(text)
    text = re.sub(r'<[^>]+>', '', text) # Remove HTML tags
    
    tokens = [t for sent in sent_tokenize(text) for t in word_tokenize(sent)]
    
    tagged = nltk.pos_tag(tokens)
    lemmas = []
    for word, tag in tagged:
        word = word.lower()
        if word in STOPWORDS or not word.isalnum():
            continue
            
        wn_tag = _penntag_to_wordnet(tag)
        if wn_tag:
            lemma = lemmatizer.lemmatize(word, wn_tag)
        else:
            lemma = word
        lemmas.append(lemma)
        
    return lemmas

# ==========================================
# 4. HELPER METRICS
# ==========================================

def _jaccard(set1, set2):
    """Computes Jaccard Similarity: |A ∩ B| / |A ∪ B|"""
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)

# ==========================================
# 5. FEATURE EXTRACTION FUNCTIONS
# ==========================================

def lexical_features(s1: str, s2: str) -> np.ndarray:
    """
    Extracts surface-level Lexical features.
    
    Features:
    1. Jaccard Similarity (Lemmatized)
    2. Overlap Coefficient
    3. Normalized Length Difference
    """
    lem1 = set(preprocess(s1))
    lem2 = set(preprocess(s2))
    len1, len2 = len(lem1), len(lem2)
    
    # 1. Jaccard
    jac = _jaccard(lem1, lem2)
    
    # 2. Overlap Coefficient (Matches subset capability)
    denom = min(len1, len2)
    over = len(lem1 & lem2) / denom if denom > 0 else 0.0

    # 3. Length Difference
    if max(len1, len2) > 0:
        ldiff = abs(len1 - len2) / max(len1, len2)
    else:
        ldiff = 0.0
    
    return np.array([jac, over, ldiff])

def semantic_features(s1: str, s2: str) -> np.ndarray:
    """
    Extracts Semantic features using WordNet.
    
    INSPIRATION:
    TakeLab (Section 2.1) utilizes Path Length Similarity from WordNet to capture 
    synonyms (e.g., Car ~ Automobile).
    
    Features:
    1. Average Path Similarity
    2. Max Path Similarity
    """
    # Use raw tokenization to keep POS tags aligned for WordNet
    t1 = nltk.pos_tag(word_tokenize(s1))
    t2 = nltk.pos_tag(word_tokenize(s2))
    
    # Get Synsets for content words
    synsets1 = [wn.synsets(w, _penntag_to_wordnet(t))[0] 
                for w, t in t1 if _penntag_to_wordnet(t) and wn.synsets(w, _penntag_to_wordnet(t))]
    synsets2 = [wn.synsets(w, _penntag_to_wordnet(t))[0] 
                for w, t in t2 if _penntag_to_wordnet(t) and wn.synsets(w, _penntag_to_wordnet(t))]

    if not synsets1 or not synsets2:
        return np.array([0.0, 0.0])

    # S1 -> S2 Best Match
    scores1 = []
    for s1_syn in synsets1:
        # Find best match in S2 for this word in S1
        best = max([s1_syn.path_similarity(s2_syn) or 0 for s2_syn in synsets2] + [0])
        scores1.append(best)
        
    # S2 -> S1 Best Match
    scores2 = []
    for s2_syn in synsets2:
        best = max([s2_syn.path_similarity(s1_syn) or 0 for s1_syn in synsets1] + [0])
        scores2.append(best)
        
    avg_sim = (sum(scores1) + sum(scores2)) / (len(scores1) + len(scores2))
    max_sim = max(scores1 + scores2 + [0])
    
    return np.array([avg_sim, max_sim])

def syntactic_features(s1: str, s2: str) -> np.ndarray:
    """
    Extracts Syntactic features using Dependency Parsing (spaCy).
    
    INSPIRATION:
    TakeLab (Section 3.4) emphasizes "Syntactic Roles Similarity", explicitly 
    comparing Predicates, Subjects, and Objects.
    
    Features:
    1. Root Verb Match (Exact Match)
    2. Subject Match (Jaccard)
    3. Object Match (Jaccard)
    """
    if not nlp: return np.array([0.0, 0.0, 0.0])
    
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    
    # 1. Root Verb (Predicate)
    root1 = {t.lemma_ for t in doc1 if t.dep_ == 'ROOT'}
    root2 = {t.lemma_ for t in doc2 if t.dep_ == 'ROOT'}
    root_match = 1.0 if not root1.isdisjoint(root2) else 0.0
    
    # 2. Subject (nsubj)
    subj1 = {t.lemma_ for t in doc1 if 'subj' in t.dep_}
    subj2 = {t.lemma_ for t in doc2 if 'subj' in t.dep_}
    subj_jac = _jaccard(subj1, subj2)
    
    # 3. Object (dobj/pobj)
    obj1 = {t.lemma_ for t in doc1 if 'obj' in t.dep_}
    obj2 = {t.lemma_ for t in doc2 if 'obj' in t.dep_}
    obj_jac = _jaccard(obj1, obj2)
    
    return np.array([root_match, subj_jac, obj_jac])

# ==========================================
# 6. DOMAIN-SPECIFIC BOOSTERS
# ==========================================

def negation_feature(s1: str, s2: str) -> float:
    """
    Checks for mismatch in negation words (did vs did NOT).
    Crucial for MSRpar (News) to distinguish opposite meanings.
    """
    negations = {"not", "no", "never", "n't", "none", "neither", "nor"}
    toks1 = set(word_tokenize(s1.lower()))
    toks2 = set(word_tokenize(s2.lower()))
    
    has_neg1 = not toks1.isdisjoint(negations)
    has_neg2 = not toks2.isdisjoint(negations)
    
    # Return 1.0 if MISMATCH (Bad), 0.0 if Match
    return 1.0 if has_neg1 != has_neg2 else 0.0

def ngram_features(s1: str, s2: str, n=2) -> float:
    """
    Jaccard similarity of N-grams to capture local context.
    INSPIRATION: UKP Lab (Section 2.1) uses "Character/word n-grams".
    """
    tokens1 = preprocess(s1)
    tokens2 = preprocess(s2)
    
    ng1 = set(ngrams(tokens1, n))
    ng2 = set(ngrams(tokens2, n))
    
    return _jaccard(ng1, ng2)

def entity_feature(s1: str, s2: str) -> float:
    """
    Named Entity Matching (spaCy).
    Ensures specific Proper Nouns (Google vs Microsoft) match.
    INSPIRATION: TakeLab (Section 3.5) uses "Named Entity Features".
    """
    if not nlp: return 0.0
    
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    
    ents1 = {e.text.lower() for e in doc1.ents}
    ents2 = {e.text.lower() for e in doc2.ents}
    
    if not ents1 and not ents2: return 0.0
    return _jaccard(ents1, ents2)

def translation_features(s1: str, s2: str) -> float:
    """
    BLEU Score: The standard metric for SMT datasets.
    INSPIRATION: Task 6 Description identifies datasets as Machine Translation outputs.
    """
    tok1 = preprocess(s1)
    tok2 = preprocess(s2)
    if not tok1 or not tok2: return 0.0
    
    cc = SmoothingFunction()
    return sentence_bleu([tok1], tok2, smoothing_function=cc.method1)

def sequence_features(s1: str, s2: str) -> float:
    """
    Longest Common Subsequence (LCS).
    INSPIRATION: UKP Lab (Section 2.1) uses "Longest common subsequence".
    """
    return SequenceMatcher(None, s1, s2).ratio()

def number_features(s1: str, s2: str) -> float:
    """
    Number Matching.
    INSPIRATION: TakeLab (Section 3.5) "Numbers Overlap" to improve MSRpar.
    """
    nums1 = set(re.findall(r'\d+(?:[\.,]\d+)?', s1))
    nums2 = set(re.findall(r'\d+(?:[\.,]\d+)?', s2))
    
    if not nums1 and not nums2: return 1.0 # Agree on "no numbers"
    if not nums1 or not nums2: return 0.0  # Disagree
    
    return _jaccard(nums1, nums2)

def stopword_ngrams(s1: str, s2: str, n=3) -> float:
    """
    Structural Similarity via Stopwords.
    INSPIRATION: UKP Lab (Section 2.4) "Structural similarity... by computing stopword n-grams".
    """
    tok1 = [w.lower() for w in word_tokenize(s1) if w.lower() in STOPWORDS]
    tok2 = [w.lower() for w in word_tokenize(s2) if w.lower() in STOPWORDS]
    
    ng1 = set(ngrams(tok1, n))
    ng2 = set(ngrams(tok2, n))
    
    if not ng1 and not ng2: return 1.0
    if not ng1 or not ng2: return 0.0
    
    return len(ng1 & ng2) / len(ng1 | ng2)

def combined_features(s1: str, s2: str) -> np.ndarray:
    """
    The Ensemble Vector.
    Combines Lexical, Semantic, Syntactic, and Domain-Specific features.
    """
    # 1. Core Layers
    f_lex = lexical_features(s1, s2)   # 3 dims
    f_sem = semantic_features(s1, s2)  # 2 dims
    f_syn = syntactic_features(s1, s2) # 3 dims
    
    # 2. Boosters (MSRpar/OnWN)
    f_neg = np.array([negation_feature(s1, s2)])
    f_bi = np.array([ngram_features(s1, s2, n=2)])
    f_ent = np.array([entity_feature(s1, s2)])

    # 3. Domain Specifics (SMT/MSRpar)
    f_bleu = np.array([translation_features(s1, s2)])
    f_lcs = np.array([sequence_features(s1, s2)])
    f_num = np.array([number_features(s1, s2)])
    
    # 4. Style (SMT)
    f_stop = np.array([stopword_ngrams(s1, s2, n=3)])

    return np.concatenate([
        f_lex, f_sem, f_syn, 
        f_neg, f_bi, f_ent, 
        f_bleu, f_lcs, f_num, f_stop
    ])