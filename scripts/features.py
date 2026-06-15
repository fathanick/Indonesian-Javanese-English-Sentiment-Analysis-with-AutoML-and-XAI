"""
features.py
===========
Feature extraction pipeline for IJE Code-Mixed Sentiment Analysis.

Produces three feature sets:
  1. tfidf    — TF-IDF word + char n-grams + text statistics
  2. tfidf_cm — tfidf + code-mixing features (CMI, language ratios, switch points)
  3. full     — tfidf_cm + multilingual sentence embeddings

Usage:
    python features.py          # extracts all three feature sets and saves to outputs/
    from features import build_feature_sets   # programmatic use
"""

import json
import pathlib
import pickle
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUTS_DIR = SCRIPT_DIR.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Language word lists (lightweight dictionary-based language ID)
# ---------------------------------------------------------------------------
# Core Indonesian stopwords + common sentiment words
INDONESIAN_WORDS = set("""
aku kamu dia kita mereka itu ini yang dan atau tapi karena jadi sudah
akan ada bisa mau tidak bukan jangan saja lagi masih pun paling sangat
bagus baik buruk jelek senang sedih marah takut cinta suka benci malu
banget sekali lumayan cukup terlalu agak memang benar salah iya ya tidak
dengan untuk dari ke di pada oleh tentang seperti jika kalau meski
walaupun namun tetapi selain terima kasih tolong mohon selamat
positif negatif netral bagaimana kenapa kapan siapa dimana kemana
sini sana situ kami anda beliau mereka nya ku mu
mahal murah enak nikmat lezat mantap keren kece asik asyik parah gila
ngaco rusak rewel ribet susah sulit berat ringan gampang mudah puas
kecewa bangga malu capek lelah bosen bosan kesel kesal dongkol semangat
males malas ogah nggak gak ndak enggak gakpapa gapapa
bgt jg yg tp krn dgn jgn
harga biaya bayar beli jual baru lama cocok pas tepat worth
dapat dapet perlu harus wajib boleh butuh minta bantu
share like follow komen komentar pusing
lg lgi lgsg udh udah dah blm blum
ramah rajin hemat irit laris terkenal terpercaya
""".split())

# Core Javanese words
JAVANESE_WORDS = set("""
aku kowe dheweke awake dhewe iku kuwi iki sing lan utawa nanging amarga
dadi wis bakal ana bisa gelem ora dudu aja wae maneh isih pon
apik becik elek ala seneng susah nesu wedi tresna gela isin
pisan cukup rada pancen bener kliru
karo kanggo saka menyang ing marang babagan kaya yen
sanajan nanging malah kajaba matur nuwun tulung punten sugeng
kene kono kana kula panjenengan piyambakipun sampeyan
mosok emang wes uwes iso ngono ngene ngono ngendi piye
""".split())

# English words (common ones in code-mixed tweets)
ENGLISH_WORDS = set("""
i you he she it we they this that the a an and or but because so
am is are was were be been have has had do does did will would could
should may might shall can get got make made go went come came
good bad great nice amazing awesome terrible horrible
love hate like dislike happy sad angry scared
very really so too much quite enough actually
with for from to at by about if when while
though although however but still yet also even
thanks sorry please okay ok yes no not
match fix run play try use need want feel think know see look help
call check give take send show turn start stop post
plis pls plz deal free best
omg wtf lol lmao haha hehe yeah yep nope nah sure
follow comment worth limited
""".split())

# Words that are explicitly "other": platform-specific terms, internet expressions,
# laughter tokens, and anything not belonging to a single language.
OTHER_WORDS = set("""
rt .rt dm retweet quote tweet trending hastag
wkwk wkwkwk wkwkwk wkkwk wkwkw hahaha hehehe xixi xixi
lol lmao omg wtf brb afk irl smh fyi asap
""".split())

# English / loanword stems that commonly take Indonesian suffixes (-nya, -kan, etc.)
# When the stem is found here + an Indonesian suffix, the word is intra-word mixed.
_LOANWORD_STEMS_FOR_ID_SUFFIX = set("""
vibe vibes game games brand brands trend trends style styles
love hype online offline cafe cafes shop shopping
""".split())

# Indonesian stems that commonly take Javanese possessive/demonstrative suffixes
# (e.g., "kamera" + "-ne" → "kamerane"; "harga" + "-ne" → "hargane")
# Words of this form are intra-word code-mixed and should be labeled "other".
_INDONESIAN_STEMS_FOR_JV_SUFFIX = set("""
kamera foto video suara nama harga uang teman pacar baju meja kursi
rumah mobil motor hp handphone laptop layar kaca pintu jalan
barang produk obat makanan minuman tempat waktu hari bulan tahun
""".split())


def identify_word_language(word: str) -> str:
    """Dictionary + heuristic language identification for a single word.

    Priority order:
      1. OTHER_WORDS (platform terms, internet expressions) → "other"
      2. ENGLISH_WORDS explicit dict → "en"
      3. JAVANESE_WORDS explicit dict → "jv"
      4. INDONESIAN_WORDS explicit dict → "id"
      5. Loanword stem + Indonesian suffix (vibesnya, gamingnya) → "other"
      6. Indonesian morphological heuristics (-nya, me-, ber-, …) → "id"
      7. Indonesian stem + Javanese suffix (kamerane, hargane) → "other"
      8. Javanese suffix heuristic (-e, -ne, …) → "jv"
      9. Fallback → "other"
    """
    w = word.lower().strip(".,!?;:'\"()[]{}#@")
    if not w:
        return "other"
    # 1. Platform / internet expressions — always "other"
    if w in OTHER_WORDS:
        return "other"
    # 2–4. Explicit dictionaries
    if w in ENGLISH_WORDS:
        return "en"
    if w in JAVANESE_WORDS:
        return "jv"
    if w in INDONESIAN_WORDS:
        return "id"
    # 5. Loanword/English stem + Indonesian suffix → intra-word mixed
    id_suffixes = ("nya", "kan", "lah", "pun", "kah")
    for suffix in id_suffixes:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            stem = w[: -len(suffix)]
            if stem in _LOANWORD_STEMS_FOR_ID_SUFFIX:
                return "mixed"
    # 6. Indonesian morphological heuristics
    if re.search(r"(nya|kan|lah|pun|kah)$", w) or re.match(r"^(me|ber|ke|ter|di|pe)", w):
        return "id"
    # 7. Indonesian stem + Javanese suffix → intra-word mixed
    for suffix in ("ne", "e", "ku", "mu", "ake", "aken"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            stem = w[: -len(suffix)]
            if stem in _INDONESIAN_STEMS_FOR_JV_SUFFIX or stem in INDONESIAN_WORDS:
                return "mixed"
    # 8. Javanese suffix heuristic
    if re.search(r"(e|ne|ku|mu|ake|aken)$", w) and len(w) > 3:
        return "jv"
    return "other"


def compute_language_ratios(text: str) -> dict:
    """Compute per-language word ratios and code-mixing metrics for a tweet."""
    words = text.split()
    n = len(words)
    if n == 0:
        return {
            "n_words": 0,
            "ratio_id": 0.0,
            "ratio_jv": 0.0,
            "ratio_en": 0.0,
            "ratio_other": 0.0,
            "cmi": 0.0,
            "n_switch_points": 0,
            "dominant_lang_id": 0,
            "dominant_lang_jv": 0,
            "dominant_lang_en": 0,
        }

    lang_seq = [identify_word_language(w) for w in words]
    counts = defaultdict(int)
    for lang in lang_seq:
        counts[lang] += 1

    ratio_id = counts["id"] / n
    ratio_jv = counts["jv"] / n
    ratio_en = counts["en"] / n
    ratio_other = counts["other"] / n

    # Code Mixing Index (CMI): fraction of words NOT in dominant language
    # CMI = (n - max_lang_count) / n  (range [0, 1])
    max_count = max(counts.values()) if counts else 0
    cmi = (n - max_count) / n if n > 0 else 0.0

    # Switch points: consecutive words with different identified languages
    # (ignoring 'other' transitions to reduce noise)
    switch_points = 0
    prev_lang = None
    for lang in lang_seq:
        if lang == "other":
            continue
        if prev_lang is not None and lang != prev_lang:
            switch_points += 1
        prev_lang = lang

    # Dominant language as one-hot
    main_counts = {k: v for k, v in counts.items() if k != "other"}
    dominant = max(main_counts, key=main_counts.get) if main_counts else "other"

    return {
        "n_words": n,
        "ratio_id": ratio_id,
        "ratio_jv": ratio_jv,
        "ratio_en": ratio_en,
        "ratio_other": ratio_other,
        "cmi": cmi,
        "n_switch_points": switch_points,
        "dominant_lang_id": int(dominant == "id"),
        "dominant_lang_jv": int(dominant == "jv"),
        "dominant_lang_en": int(dominant == "en"),
    }


# ---------------------------------------------------------------------------
# Text statistics
# ---------------------------------------------------------------------------

def compute_text_stats(text: str) -> dict:
    """Compute simple text statistics."""
    words = text.split()
    n_words = len(words)
    n_chars = len(text)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    n_punct = sum(1 for c in text if c in ".,!?;:'\"-()[]{}")
    punct_density = n_punct / max(n_chars, 1)
    n_digits = sum(1 for c in text if c.isdigit())
    n_upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)

    return {
        "n_words": n_words,
        "n_chars": n_chars,
        "avg_word_len": avg_word_len,
        "n_punct": n_punct,
        "punct_density": punct_density,
        "n_digits": n_digits,
        "n_upper_words": n_upper_words,
    }


# ---------------------------------------------------------------------------
# Feature Extractors
# ---------------------------------------------------------------------------

class TFIDFFeatureExtractor:
    """Fit TF-IDF word + char n-grams and text stats on training data."""

    def __init__(
        self,
        word_ngram_range=(1, 3),
        char_ngram_range=(2, 4),
        max_word_features=10000,
        max_char_features=10000,
    ):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler

        self.word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram_range,
            max_features=max_word_features,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram_range,
            max_features=max_char_features,
            sublinear_tf=True,
            min_df=2,
        )
        self.stat_scaler = StandardScaler()
        self._stat_cols = None

    def fit(self, texts):
        self.word_vec.fit(texts)
        self.char_vec.fit(texts)
        stats = pd.DataFrame([compute_text_stats(t) for t in texts])
        self._stat_cols = stats.columns.tolist()
        self.stat_scaler.fit(stats.values)
        return self

    def transform(self, texts):
        word_mat = self.word_vec.transform(texts)
        char_mat = self.char_vec.transform(texts)
        stats = pd.DataFrame([compute_text_stats(t) for t in texts])
        stat_mat = self.stat_scaler.transform(stats[self._stat_cols].values)
        from scipy.sparse import csr_matrix
        return hstack([word_mat, char_mat, csr_matrix(stat_mat)])

    def get_feature_names(self):
        word_names = [f"word_{n}" for n in self.word_vec.get_feature_names_out()]
        char_names = [f"char_{n}" for n in self.char_vec.get_feature_names_out()]
        stat_names = [f"stat_{c}" for c in self._stat_cols]
        return word_names + char_names + stat_names

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


class CodeMixingFeatureExtractor:
    """Extract code-mixing features and scale them."""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self._cols = None

    def _extract(self, texts):
        rows = []
        for t in texts:
            lang = compute_language_ratios(t)
            rows.append(lang)
        return pd.DataFrame(rows)

    def fit(self, texts):
        df = self._extract(texts)
        self._cols = df.columns.tolist()
        self.scaler.fit(df.values)
        return self

    def transform(self, texts):
        from scipy.sparse import csr_matrix
        df = self._extract(texts)
        scaled = self.scaler.transform(df[self._cols].values)
        return csr_matrix(scaled)

    def get_feature_names(self):
        return [f"cm_{c}" for c in self._cols]

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


class EmbeddingFeatureExtractor:
    """Sentence embeddings from a multilingual MiniLM model."""

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self):
        self._model = None
        self.scaler = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from sklearn.preprocessing import StandardScaler
            print(f"  Loading sentence-transformer: {self.MODEL_NAME}")
            self._model = SentenceTransformer(self.MODEL_NAME)
            self.scaler = StandardScaler()

    def fit(self, texts):
        self._load_model()
        embs = self._model.encode(list(texts), batch_size=64, show_progress_bar=False)
        self.scaler.fit(embs)
        return self

    def transform(self, texts):
        from scipy.sparse import csr_matrix
        self._load_model()
        embs = self._model.encode(list(texts), batch_size=64, show_progress_bar=False)
        scaled = self.scaler.transform(embs)
        return csr_matrix(scaled)

    def get_feature_names(self):
        # We don't know dim until model is loaded; return placeholder
        dim = self._model.get_sentence_embedding_dimension() if self._model else 384
        return [f"emb_{i}" for i in range(dim)]

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


# ---------------------------------------------------------------------------
# Build all feature sets
# ---------------------------------------------------------------------------

def build_feature_sets(splits: dict[str, pd.DataFrame]):
    """
    Build three feature sets for train / valid / test.

    Returns a dict:
        {
            "tfidf":    {"train": X_sparse, "valid": ..., "test": ...},
            "tfidf_cm": {"train": X_sparse, "valid": ..., "test": ...},
            "full":     {"train": X_sparse, "valid": ..., "test": ...},
            "feature_names": {"tfidf": [...], "tfidf_cm": [...], "full": [...]},
            "extractors": {...},
        }
    """
    train_texts = splits["train"]["text"].tolist()
    valid_texts = splits["valid"]["text"].tolist()
    test_texts = splits["test"]["text"].tolist()

    # ------------------------------------------------------------------
    print("\n[Feature Extraction] Fitting TF-IDF extractor on train…")
    tfidf_ext = TFIDFFeatureExtractor()
    X_train_tfidf = tfidf_ext.fit_transform(train_texts)
    X_valid_tfidf = tfidf_ext.transform(valid_texts)
    X_test_tfidf = tfidf_ext.transform(test_texts)
    print(f"  TF-IDF shape: {X_train_tfidf.shape}")

    # ------------------------------------------------------------------
    print("\n[Feature Extraction] Fitting Code-Mixing extractor on train…")
    cm_ext = CodeMixingFeatureExtractor()
    X_train_cm = cm_ext.fit_transform(train_texts)
    X_valid_cm = cm_ext.transform(valid_texts)
    X_test_cm = cm_ext.transform(test_texts)
    print(f"  CM feature shape: {X_train_cm.shape}")

    # Combine: TF-IDF + CM
    X_train_tfidf_cm = hstack([X_train_tfidf, X_train_cm])
    X_valid_tfidf_cm = hstack([X_valid_tfidf, X_valid_cm])
    X_test_tfidf_cm = hstack([X_test_tfidf, X_test_cm])
    print(f"  TF-IDF+CM shape: {X_train_tfidf_cm.shape}")

    # ------------------------------------------------------------------
    print("\n[Feature Extraction] Fitting Embedding extractor on train…")
    emb_ext = EmbeddingFeatureExtractor()
    X_train_emb = emb_ext.fit_transform(train_texts)
    X_valid_emb = emb_ext.transform(valid_texts)
    X_test_emb = emb_ext.transform(test_texts)
    print(f"  Embedding shape: {X_train_emb.shape}")

    # Combine: TF-IDF + CM + Embeddings
    X_train_full = hstack([X_train_tfidf_cm, X_train_emb])
    X_valid_full = hstack([X_valid_tfidf_cm, X_valid_emb])
    X_test_full = hstack([X_test_tfidf_cm, X_test_emb])
    print(f"  Full feature shape: {X_train_full.shape}")

    # ------------------------------------------------------------------
    feature_names = {
        "tfidf": tfidf_ext.get_feature_names(),
        "tfidf_cm": tfidf_ext.get_feature_names() + cm_ext.get_feature_names(),
        "full": tfidf_ext.get_feature_names() + cm_ext.get_feature_names() + emb_ext.get_feature_names(),
    }

    return {
        "tfidf": {"train": X_train_tfidf, "valid": X_valid_tfidf, "test": X_test_tfidf},
        "tfidf_cm": {"train": X_train_tfidf_cm, "valid": X_valid_tfidf_cm, "test": X_test_tfidf_cm},
        "full": {"train": X_train_full, "valid": X_valid_full, "test": X_test_full},
        "feature_names": feature_names,
        "extractors": {
            "tfidf": tfidf_ext,
            "cm": cm_ext,
            "emb": emb_ext,
        },
    }


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_feature_sets(feature_sets: dict, out_dir: pathlib.Path = OUTPUTS_DIR):
    """Save feature matrices and extractors to disk."""
    import scipy.sparse as sp

    out_dir.mkdir(parents=True, exist_ok=True)
    for fs_name in ["tfidf", "tfidf_cm", "full"]:
        for split in ["train", "valid", "test"]:
            mat = feature_sets[fs_name][split]
            path = out_dir / f"X_{fs_name}_{split}.npz"
            sp.save_npz(str(path), mat)

    # Feature names
    names_path = out_dir / "feature_names.json"
    names_path.write_text(
        json.dumps(feature_sets["feature_names"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Extractors
    ext_path = out_dir / "extractors.pkl"
    with open(ext_path, "wb") as f:
        pickle.dump(feature_sets["extractors"], f)

    print(f"\nFeature sets saved to {out_dir}")


def load_feature_sets(out_dir: pathlib.Path = OUTPUTS_DIR) -> dict:
    """Load previously saved feature sets from disk."""
    import scipy.sparse as sp

    feature_sets: dict = {"tfidf": {}, "tfidf_cm": {}, "full": {}}
    for fs_name in ["tfidf", "tfidf_cm", "full"]:
        for split in ["train", "valid", "test"]:
            path = out_dir / f"X_{fs_name}_{split}.npz"
            feature_sets[fs_name][split] = sp.load_npz(str(path))

    names_path = out_dir / "feature_names.json"
    feature_sets["feature_names"] = json.loads(names_path.read_text(encoding="utf-8"))

    ext_path = out_dir / "extractors.pkl"
    with open(ext_path, "rb") as f:
        feature_sets["extractors"] = pickle.load(f)

    return feature_sets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import sys
    sys.path.insert(0, str(SCRIPT_DIR))
    from preprocess import main as preprocess_main, load_data

    # Load pre-processed data (run preprocess.py first if needed)
    try:
        splits = load_data()
        print("Loaded pre-processed data from disk.")
    except FileNotFoundError:
        print("Pre-processed data not found — running preprocess.py first.")
        splits = preprocess_main()

    # Labels
    for split_name, df in splits.items():
        print(f"  {split_name}: {len(df)} samples")

    feature_sets = build_feature_sets(splits)

    # Save labels as well
    for split_name, df in splits.items():
        y_path = OUTPUTS_DIR / f"y_{split_name}.npy"
        np.save(str(y_path), df["label_id"].values)

    save_feature_sets(feature_sets)

    # Summary
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    for fs_name in ["tfidf", "tfidf_cm", "full"]:
        shape = feature_sets[fs_name]["train"].shape
        print(f"  {fs_name:12s}: {shape[0]} samples × {shape[1]} features")
    print("=" * 60)


if __name__ == "__main__":
    main()
