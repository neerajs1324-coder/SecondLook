"""Turns a message into tokens and red flags.

web/scorer.js is a line for line port of this file, and tests/check_parity.mjs proves the
two give identical scores, so the browser reads messages exactly the way the model was trained.
"""

import json
import re
import unicodedata
from pathlib import Path

RULES = json.loads((Path(__file__).resolve().parent.parent / "web" / "rules.json").read_text())

FLAG_NAMES = [
    "url_present", "url_shortener", "suspicious_tld", "brand_lookalike", "ip_url",
    *RULES["flags"].keys(),
    "shouting", "has_money", "has_phone",
]

A = re.ASCII
STRONG_TLDS = [t for t in RULES["tlds"] if t not in {"me", "in", "at", "be", "co", "us", "gd", "gl", "cc", "tv", "ca", "au", "de", "fr", "eu", "mobi", "ly"}]
_shorteners = "|".join(re.escape(s) for s in sorted(RULES["shorteners"], key=len, reverse=True))
_tlds = "|".join(sorted(STRONG_TLDS, key=len, reverse=True))

EMAIL_RE = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", A)
URL_RE = re.compile(
    r"https?://[^\s]+|www\.[^\s]+|\b(?:" + _shorteners + r")(?:/[^\s]*)?|\b(?:[a-z0-9-]+\.)+(?:" + _tlds + r")\b(?:/[^\s]*)?",
    A,
)
MONEY_RE = re.compile(r"(?:[£$€₹]|\brs\.?|\binr|\busd)\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:dollars|pounds|usd|gbp)\b", A)
PHONE_RE = re.compile(r"\+?\d(?:[\s().-]?\d){6,}", A)
NUM_RE = re.compile(r"\d+", A)
TOKEN_RE = re.compile(r"__[a-z]+__|[a-z]+", A)
IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$", A)
FLAG_RES = {name: re.compile(p, A) for name, p in RULES["flags"].items()}
BRAND_FILLER = "|".join(RULES["brandFillers"])


def domain_of(url):
    d = re.sub(r"^https?://", "", url)
    d = re.split(r"[/?#:]", d)[0]
    return re.sub(r"^www\.", "", d)


def brand_lookalike(domain):
    """True when a domain borrows a brand name, like usps-redelivery.top, but isn't the real site."""
    labels = re.split(r"[.-]", domain)
    for brand, official in RULES["brands"].items():
        if any(domain == o or domain.endswith("." + o) for o in official):
            continue
        pattern = re.compile(r"^(?:(?:" + BRAND_FILLER + r")?" + brand + r"(?:" + BRAND_FILLER + r")?)$", A)
        if any(pattern.match(label) for label in labels):
            return True
    return False


def analyze(text):
    """Returns (tokens, flags dict, list of link domains)."""
    text = unicodedata.normalize("NFKC", text)
    letters = [c for c in text if c.isascii() and c.isalpha()]
    upper = sum(1 for c in letters if c.isupper())
    lower = text.lower()

    flags = dict.fromkeys(FLAG_NAMES, 0)
    flags["shouting"] = int(len(letters) >= 12 and upper / len(letters) >= 0.4)

    s = EMAIL_RE.sub(" __email__ ", lower)
    domains = []
    for m in URL_RE.finditer(s):
        url = m.group(0).rstrip(".,!?)'\"")
        d = domain_of(url)
        domains.append(d)
        flags["url_present"] = 1
        if d in RULES["shorteners"]:
            flags["url_shortener"] = 1
        if d.split(".")[-1] in RULES["suspiciousTlds"]:
            flags["suspicious_tld"] = 1
        if IP_RE.match(d):
            flags["ip_url"] = 1
        if brand_lookalike(d):
            flags["brand_lookalike"] = 1
    s = URL_RE.sub(" __url__ ", s)

    for name, rx in FLAG_RES.items():
        flags[name] = int(bool(rx.search(s)))

    if MONEY_RE.search(s):
        flags["has_money"] = 1
    s = MONEY_RE.sub(" __money__ ", s)
    if PHONE_RE.search(s):
        flags["has_phone"] = 1
    s = PHONE_RE.sub(" __phone__ ", s)
    s = NUM_RE.sub(" __num__ ", s)

    return TOKEN_RE.findall(s), flags, domains


def ngrams(tokens):
    return tokens + [tokens[i] + " " + tokens[i + 1] for i in range(len(tokens) - 1)]
