import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from features import analyze, brand_lookalike  # noqa: E402


def on(text):
    return {k for k, v in analyze(text)[1].items() if v}


def test_fake_brand_domains_are_caught():
    assert brand_lookalike("usps-redelivery.top")
    assert brand_lookalike("paypal-refunds.online")
    assert brand_lookalike("ezpass-tolls.com")


def test_real_brand_domains_are_not_flagged():
    assert not brand_lookalike("tools.usps.com")
    assert not brand_lookalike("www.paypal.com".removeprefix("www."))
    assert not brand_lookalike("groups.google.com")  # contains "ups" but isn't pretending to be UPS
    assert not brand_lookalike("applebees.com")


def test_link_warning_signs():
    assert {"url_shortener", "url_present"} <= on("verify at bit.ly/abc123")
    assert "suspicious_tld" in on("go to claim-now.xyz today")
    assert "ip_url" in on("login at http://185.62.48.11/bank")


def test_message_warning_signs():
    assert "payment_ask" in on("pay the fee with google play gift cards")
    assert "account_problem" in on("your account has been suspended")
    assert "delivery_issue" in on("your package is on hold, update your address")
    assert "unpaid_bill" in on("you have an unpaid toll balance")
    assert "shouting" in on("URGENT ACTION REQUIRED ON YOUR ACCOUNT")


def test_normal_texts_stay_clean():
    assert on("hey are we still getting food after practice?") == set()
    assert "url_present" not in on("i will.be there at 5")


def test_numbers_are_masked_so_digits_never_become_words():
    tokens = analyze("call 248-533-7085 about the $20 and room 5")[0]
    assert "__phone__" in tokens and "__money__" in tokens and "__num__" in tokens
    assert not any(t.isdigit() for t in tokens)
