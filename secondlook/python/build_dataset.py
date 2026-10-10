"""Merges the two public SMS datasets, removes copies, and splits by campaign.

Outputs data/corpus.csv with columns: text, label (legit / spam / scam), source, group, split.
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import ftfy
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = {"ham": "legit", "spam": "spam", "smishing": "scam"}
SEVERITY = {"legit": 0, "spam": 1, "scam": 2}


def clean(text):
    return re.sub(r"\s+", " ", ftfy.fix_text(text)).strip()


def key(text):
    """Letters and digits only, so encoding junk and spacing can't hide a duplicate."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def load():
    messages = {}  # key -> dict
    stats = Counter()

    with open(DATA / "sms_phishing_mendeley.csv", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            text, label = clean(row["TEXT"]), LABELS[row["LABEL"].strip().lower()]
            k = key(text)
            if not k:
                continue
            stats["mendeley_rows"] += 1
            if k in messages:
                stats["mendeley_duplicates"] += 1
                # the same message labeled two ways keeps the more severe label
                if SEVERITY[label] > SEVERITY[messages[k]["label"]]:
                    messages[k]["label"] = label
                continue
            messages[k] = {"text": text, "label": label, "source": "mendeley"}

    with open(DATA / "sms_uci.tsv", encoding="utf-8", errors="replace") as f:
        for line in f:
            label, _, text = line.rstrip("\n").partition("\t")
            text = clean(text)
            k = key(text)
            if not k:
                continue
            stats["uci_rows"] += 1
            if k in messages:
                stats["uci_already_in_mendeley" if messages[k]["source"] != "uci" else "uci_duplicates"] += 1
                if messages[k]["source"] == "mendeley":
                    messages[k]["source"] = "both"
                continue
            # UCI only says "spam", it never says whether a spam message is a scam,
            # so spam that Mendeley didn't relabel can't be used for training
            if label == "spam":
                messages[k] = {"text": text, "label": "spam_unknown", "source": "uci"}
            else:
                messages[k] = {"text": text, "label": "legit", "source": "uci"}

    return list(messages.values()), stats


def campaign_groups(rows):
    """Clusters near copies (same template, different numbers) so they stay in one split."""
    lsh = MinHashLSH(threshold=0.7, num_perm=128)
    hashes = []
    for i, r in enumerate(rows):
        t = re.sub(r"\d", "0", r["text"].lower())
        m = MinHash(num_perm=128)
        for j in range(max(1, len(t) - 4)):
            m.update(t[j:j + 5].encode())
        hashes.append(m)
        lsh.insert(str(i), m)

    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, m in enumerate(hashes):
        for j in lsh.query(m):
            a, b = find(i), find(int(j))
            if a != b:
                parent[a] = b
    return [find(i) for i in range(len(rows))]


def main():
    rows, stats = load()
    groups = campaign_groups(rows)
    for r, g in zip(rows, groups):
        r["group"] = g

    trainable = [r for r in rows if r["label"] != "spam_unknown"]
    labels = [r["label"] for r in trainable]
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=7)
    folds = list(sgkf.split(trainable, labels, [r["group"] for r in trainable]))
    for idx in folds[0][1]:
        trainable[idx]["split"] = "test"
    for idx in folds[1][1]:
        trainable[idx]["split"] = "validation"
    for r in trainable:
        r.setdefault("split", "train")
    for r in rows:
        if r["label"] == "spam_unknown":
            r["split"] = "extra_check"

    with open(DATA / "corpus.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "source", "group", "split"])
        w.writeheader()
        w.writerows(rows)

    group_sizes = Counter(r["group"] for r in trainable)
    in_campaign = Counter(r["label"] for r in trainable if group_sizes[r["group"]] > 1)
    totals = Counter(r["label"] for r in trainable)
    table = defaultdict(Counter)
    for r in rows:
        table[r["split"]][r["label"]] += 1

    audit = {
        **stats,
        "unique_messages": len(rows),
        "share_in_multi_message_campaigns": {k: round(in_campaign[k] / totals[k], 3) for k in totals},
        "splits": {k: dict(v) for k, v in table.items()},
    }
    (DATA / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
