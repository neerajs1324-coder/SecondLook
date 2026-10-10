"""Trains the scam classifier, compares it with simpler baselines, and exports it for the browser.

Writes web/model.json, reports/metrics.json, and tests/fixtures.json.
"""

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FLAG_NAMES, RULES, analyze, ngrams  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["legit", "spam", "scam"]
FLAG_SCALE = 1.0
THRESHOLDS = {"scam": 0.5, "careful": 0.2, "spam": 0.5}


class Vectorizer:
    """TF-IDF over word unigrams and bigrams, written by hand so the browser can copy it exactly."""

    def __init__(self, min_df=2):
        self.min_df = min_df

    def fit(self, docs):
        df = Counter()
        for d in docs:
            df.update(set(ngrams(d[0])))
        terms = sorted(t for t, c in df.items() if c >= self.min_df)
        self.vocab = {t: i for i, t in enumerate(terms)}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in terms])
        return self

    def transform(self, docs):
        rows, cols, vals = [], [], []
        for r, (tokens, flags, _) in enumerate(docs):
            counts = Counter(g for g in ngrams(tokens) if g in self.vocab)
            weights = {self.vocab[g]: (1 + math.log(c)) * self.idf[self.vocab[g]] for g, c in counts.items()}
            norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
            for c, w in weights.items():
                rows.append(r)
                cols.append(c)
                vals.append(w / norm)
        text = csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)))
        flags = csr_matrix(np.array([[f[name] * FLAG_SCALE for name in FLAG_NAMES] for _, f, _ in docs], dtype=float))
        return hstack([text, flags]).tocsr()


def verdict(p, flags):
    """Same decision rule the website uses."""
    scam, spam = p[2], p[1]
    severe = any(flags[f] for f in RULES["severe"])
    if scam >= THRESHOLDS["scam"]:
        return "scam"
    if severe or scam >= THRESHOLDS["careful"]:
        return "careful"
    if spam >= THRESHOLDS["spam"]:
        return "spam"
    return "okay"


def warning_rates(probs, docs, labels):
    v = [verdict(p, d[1]) for p, d in zip(probs, docs)]
    warned = lambda lab: np.mean([x in ("scam", "careful") for x, l in zip(v, labels) if l == lab])
    return {
        "scams_warned_about": round(float(warned("scam")), 3),
        "legit_texts_wrongly_warned": round(float(warned("legit")), 3),
        "spam_warned_about": round(float(warned("spam")), 3),
    }


def main():
    with open(ROOT / "data" / "corpus.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split = lambda name: [r for r in rows if r["split"] == name]
    train, val, test, extra = split("train"), split("validation"), split("test"), split("extra_check")
    docs = lambda rs: [analyze(r["text"]) for r in rs]
    y = lambda rs: np.array([CLASSES.index(r["label"]) for r in rs])

    d_train, d_val, d_test, d_extra = docs(train), docs(val), docs(test), docs(extra)

    # pick regularization strength on the validation set
    best = None
    vec = Vectorizer().fit(d_train)
    for C in [0.3, 1, 3, 10, 30, 100]:
        m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
        m.fit(vec.transform(d_train), y(train))
        score = f1_score(y(val), m.predict(vec.transform(d_val)), average="macro")
        print(f"C={C:<5} validation macro F1 {score:.3f}")
        if best is None or score > best[1]:
            best = (C, score)
    C = best[0]

    # refit on train + validation, then score the untouched test set exactly once
    d_fit, y_fit = d_train + d_val, np.concatenate([y(train), y(val)])
    vec = Vectorizer().fit(d_fit)
    X_fit, X_test = vec.transform(d_fit), vec.transform(d_test)
    model = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X_fit, y_fit)
    probs = model.predict_proba(X_test)
    pred = probs.argmax(1)

    # baselines, so the final numbers mean something
    n_text = len(vec.vocab)
    nb = MultinomialNB().fit(X_fit[:, :n_text], y_fit)
    rules_only = LogisticRegression(max_iter=5000, class_weight="balanced").fit(X_fit[:, n_text:], y_fit)
    text_only = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(X_fit[:, :n_text], y_fit)

    def summary(p):
        return {
            "macro_f1": round(f1_score(y(test), p, average="macro"), 3),
            "scam_recall": round(float(np.mean(p[y(test) == 2] == 2)), 3),
            "legit_flagged_as_scam": round(float(np.mean(p[y(test) == 0] == 2)), 3),
        }

    extra_probs = model.predict_proba(vec.transform(d_extra))
    metrics = {
        "chosen_C": C,
        "test_size": dict(Counter(r["label"] for r in test)),
        "model": summary(pred),
        "baselines": {
            "naive_bayes_words_only": summary(nb.predict(X_test[:, :n_text])),
            "logistic_words_only": summary(text_only.predict(X_test[:, :n_text])),
            "red_flags_only": summary(rules_only.predict(X_test[:, n_text:])),
        },
        "website_verdicts": warning_rates(probs, d_test, [r["label"] for r in test]),
        "old_uci_spam_not_marked_okay": round(float(np.mean([verdict(p, d[1]) != "okay" for p, d in zip(extra_probs, d_extra)])), 3),
        "confusion_matrix": {"rows_true_cols_pred": CLASSES, "values": confusion_matrix(y(test), pred).tolist()},
        "report": classification_report(y(test), pred, target_names=CLASSES, output_dict=True),
    }

    # hand written modern examples, a sanity check only, never used for training or tuning
    ex = json.loads((ROOT / "examples" / "modern_examples.json").read_text())
    d_ex = [analyze(e["text"]) for e in ex]
    p_ex = model.predict_proba(vec.transform(d_ex))
    results = []
    for e, p, d in zip(ex, p_ex, d_ex):
        results.append({"text": e["text"], "expected": e["label"], "verdict": verdict(p, d[1]), "scam_probability": round(float(p[2]), 3)})
    p_words = text_only.predict_proba(vec.transform(d_ex)[:, :n_text])
    called = lambda P, lab: f"{sum(pp[2] >= 0.5 for e, pp in zip(ex, P) if e['label'] == lab)} of {sum(e['label'] == lab for e in ex)}"
    metrics["modern_examples_model_alone"] = {
        "words_only": {"scams_called_scam": called(p_words, "scam"), "legit_called_scam": called(p_words, "legit")},
        "words_plus_red_flags": {"scams_called_scam": called(p_ex, "scam"), "legit_called_scam": called(p_ex, "legit")},
    }
    metrics["modern_examples"] = {
        "scams_warned_about": f"{sum(r['verdict'] in ('scam', 'careful') for r in results if r['expected'] == 'scam')} of {sum(r['expected'] == 'scam' for r in results)}",
        "legit_wrongly_warned": f"{sum(r['verdict'] in ('scam', 'careful') for r in results if r['expected'] == 'legit')} of {sum(r['expected'] == 'legit' for r in results)}",
        "results": results,
    }

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    terms = [None] * n_text
    for t, i in vec.vocab.items():
        terms[i] = t
    export = {
        "classes": CLASSES,
        "terms": terms,
        "idf": [round(float(v), 4) for v in vec.idf],
        "flagNames": FLAG_NAMES,
        "flagScale": FLAG_SCALE,
        "weights": [[round(float(w), 4) for w in row] for row in model.coef_],
        "intercept": [round(float(b), 5) for b in model.intercept_],
        "thresholds": THRESHOLDS,
        "trainedOn": len(d_fit),
    }
    (ROOT / "web" / "model.json").write_text(json.dumps(export, separators=(",", ":")))

    # expected browser outputs for the parity test, computed from the rounded exported weights
    W = np.array(export["weights"])
    b = np.array(export["intercept"])
    idf = np.array(export["idf"])
    vec.idf = idf
    sample = [r["text"] for r in test] + [e["text"] for e in ex]
    Xs = vec.transform([analyze(t) for t in sample]).toarray()
    logits = Xs @ W.T + b
    ps = np.exp(logits - logits.max(1, keepdims=True))
    ps /= ps.sum(1, keepdims=True)
    fixtures = [{"text": t, "probs": [round(float(v), 6) for v in p], "flags": [n for n, v in analyze(t)[1].items() if v]} for t, p in zip(sample, ps)]
    (ROOT / "tests" / "fixtures.json").write_text(json.dumps(fixtures))

    print(json.dumps({k: metrics[k] for k in ["chosen_C", "test_size", "model", "baselines", "website_verdicts", "old_uci_spam_not_marked_okay", "confusion_matrix"]}, indent=2))
    print("modern examples:", metrics["modern_examples"]["scams_warned_about"], "scams warned,",
          metrics["modern_examples"]["legit_wrongly_warned"], "legit wrongly warned")
    for r in results:
        ok = (r["verdict"] in ("scam", "careful")) == (r["expected"] == "scam")
        print(("  " if ok else "X ") + f"{r['expected']:5} -> {r['verdict']:7} {r['scam_probability']:.2f}  {r['text'][:80]}")
    print(f"exported {n_text} terms, model.json {(ROOT / 'web' / 'model.json').stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
