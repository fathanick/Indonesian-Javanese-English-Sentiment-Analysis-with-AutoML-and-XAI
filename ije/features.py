"""Complete raw-text feature pipeline; every learned transformer is fold-local."""
from functools import lru_cache
import re

LANGUAGES = ("ID", "JV", "EN", "MIX_ID_EN", "MIX_ID_JV", "MIX_JV_EN", "OTH")
SINGLE = ("ID", "JV", "EN")
MIXED = ("MIX_ID_EN", "MIX_ID_JV", "MIX_JV_EN")


def diversity(tags):
    from collections import Counter
    return 1.0 - max(Counter(tags).values()) / len(tags) if tags else 0.0


def cm_row(tags):
    """21 explicit features. Undefined single-language CMI is encoded as zero.

    cmi_single_only excludes OTH and MIX labels (0..1 scale, no x100).
    tag_diversity_all includes all seven tags, including OTH.
    tag_diversity_content excludes OTH but treats MIX labels as categories.
    These are deliberately distinct quantities, not interchangeable CMI values.
    Switching is measured over adjacent retained non-OTH tags; its rate uses
    that retained sequence's number of transitions as its denominator.
    """
    from collections import Counter
    if not set(tags) <= set(LANGUAGES):
        raise ValueError("Unexpected language tag")
    n = len(tags)
    counts = Counter(tags)
    content = [tag for tag in tags if tag != "OTH"]
    single = [tag for tag in tags if tag in SINGLE]
    switches = sum(a != b for a, b in zip(content, content[1:]))
    dominant = max(LANGUAGES, key=lambda tag: counts[tag]) if n else None
    return {
        **{f"ratio_{tag}": counts[tag] / n if n else 0.0 for tag in LANGUAGES},
        "n_words": n,
        "cmi_single_only": diversity(single),
        "tag_diversity_all": diversity(tags),
        "tag_diversity_content": diversity(content),
        "switch_count": switches,
        "switch_rate": switches / (len(content) - 1) if len(content) > 1 else 0.0,
        "ratio_mixed": sum(counts[tag] for tag in MIXED) / n if n else 0.0,
        **{f"dominant_{tag}": int(tag == dominant) for tag in LANGUAGES},
    }


def text_stats(text):
    words = text.split()
    punct = sum(c in ".,!?;:'\"-()[]{}" for c in text)
    return [len(words), len(text), sum(map(len, words)) / max(len(words), 1),
            punct, punct / max(len(text), 1), sum(c.isdigit() for c in text),
            sum(w.isupper() and len(w) > 1 for w in words)]


def _revision(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise ValueError("A fixed 40-character model revision commit is required")
    return value


@lru_cache(maxsize=2)
def lid_backend(model_name, revision, threads):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_name, revision=_revision(revision))
    model = AutoModelForTokenClassification.from_pretrained(model_name, revision=revision)
    model.to("cpu").eval()
    if set(model.config.id2label.values()) != set(LANGUAGES):
        raise ValueError("IJELID label schema differs from protocol")
    return tok, model


def language_tags(texts, config):
    """Contextual prediction, exact whitespace-token spans, CPU only.

    Replicates the supplied word/subword alignment rule. Empty perturbations
    produce no tags. Long inputs fail explicitly instead of being truncated.
    """
    import numpy as np
    import torch
    lid = config["features"]["lid"]
    tok, model = lid_backend(lid["model"], lid["revision"], config["cpu_threads"])
    rows = []
    for offset in range(0, len(texts), 32):
        batch = texts[offset:offset + 32]
        sequences, spans = [], []
        for text in batch:
            ids, word_spans = [tok.cls_token_id], []
            for word in text.split():
                sub = tok(word, add_special_tokens=False)["input_ids"] or [tok.unk_token_id]
                word_spans.append((len(ids), len(ids) + len(sub)))
                ids.extend(sub)
            ids.append(tok.sep_token_id)
            if len(ids) > min(512, model.config.max_position_embeddings):
                raise ValueError("Input exceeds IJELID positional limit; no silent truncation")
            sequences.append(ids)
            spans.append(word_spans)
        width = max(map(len, sequences))
        ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long, device="cpu")
        masks = torch.zeros_like(ids)
        for i, sequence in enumerate(sequences):
            ids[i, :len(sequence)] = torch.tensor(sequence, device="cpu")
            masks[i, :len(sequence)] = 1
        with torch.inference_mode():
            predictions = model(input_ids=ids, attention_mask=masks).logits.argmax(-1).cpu().numpy()
        for i, word_spans in enumerate(spans):
            tags = []
            for start, stop in word_spans:
                values, counts = np.unique(predictions[i, start:stop], return_counts=True)
                tags.append(model.config.id2label[int(values[counts.argmax()])])
            rows.append(tags)
    return rows


@lru_cache(maxsize=2)
def embedding_backend(model_name, revision, threads):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(threads)
    return SentenceTransformer(model_name, revision=_revision(revision), device="cpu")


class TextFeatures:
    """Serializable transforms shared by evaluation and LIME, including cleaning.

    Frozen pretrained encoders are loaded separately by immutable revision;
    vectorizers and scaling parameters are serialized with the fitted object.
    """
    def __init__(self, feature_set, config):
        if feature_set not in ("tfidf", "tfidf_cm", "full"):
            raise ValueError(feature_set)
        self.feature_set = feature_set
        self.config = config
        self.fitted = False

    def _texts(self, texts):
        from .data import clean_text
        return [clean_text(t) for t in texts]

    def _dense_blocks(self, texts):
        import numpy as np
        blocks = {"stats": np.asarray([text_stats(t) for t in texts], dtype=float)}
        if self.feature_set != "tfidf":
            blocks["cm"] = np.asarray([list(cm_row(tags).values())
                                      for tags in language_tags(texts, self.config)], dtype=float)
        if self.feature_set == "full":
            e = self.config["features"]["embedding"]
            encoder = embedding_backend(e["model"], e["revision"], self.config["cpu_threads"])
            blocks["embedding"] = encoder.encode(texts, batch_size=32, device="cpu",
                                                  show_progress_bar=False, convert_to_numpy=True)
        return blocks

    def fit_transform(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler
        from scipy.sparse import csr_matrix, hstack
        import numpy as np
        texts = self._texts(texts)
        f = self.config["features"]
        self.word = TfidfVectorizer(analyzer="word", ngram_range=tuple(f["word_ngrams"]),
                                   max_features=f["max_word_features"], min_df=f["min_df"],
                                   sublinear_tf=f["sublinear_tf"], strip_accents="unicode")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(f["char_ngrams"]),
                                   max_features=f["max_char_features"], min_df=f["min_df"],
                                   sublinear_tf=f["sublinear_tf"])
        sparse = [self.word.fit_transform(texts), self.char.fit_transform(texts)]
        blocks = self._dense_blocks(texts)
        self.scalers = {name: StandardScaler() for name in blocks}
        self.constant_dense_columns = {}
        for name, values in blocks.items():
            self.constant_dense_columns[name] = np.flatnonzero(np.var(values, axis=0) == 0).tolist()
            sparse.append(csr_matrix(self.scalers[name].fit_transform(values)))
        matrix = hstack(sparse, format="csr", dtype=np.float32)
        self.n_features = matrix.shape[1]
        self.fitted = True
        return matrix

    def transform(self, texts):
        import numpy as np
        from scipy.sparse import csr_matrix, hstack
        if not self.fitted:
            raise RuntimeError("Feature extractor has not been fitted")
        texts = self._texts(texts)
        sparse = [self.word.transform(texts), self.char.transform(texts)]
        for name, values in self._dense_blocks(texts).items():
            sparse.append(csr_matrix(self.scalers[name].transform(values)))
        matrix = hstack(sparse, format="csr", dtype=np.float32)
        if matrix.shape[1] != self.n_features:
            raise ValueError("Feature dimension changed at prediction time")
        return matrix
