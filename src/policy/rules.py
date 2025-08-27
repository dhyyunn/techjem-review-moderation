import re

URL = re.compile(r"(https?://|www\.)", re.I)
PHONE = re.compile(r"\+?\d[\d\-\s]{6,}\d")
COUPON = re.compile(r"(promo|coupon|discount|\\bdeal\\b|% ?off|use code)", re.I)
NO_VISIT = re.compile(r"(never been|haven'?t been|didn'?t go (in|inside)|did not visit|heard (that|it))", re.I)
CONTACT_ME = re.compile(r"(dm|contact me|whats?app|telegram|kik|line)\\b", re.I)
GIBBERISH = re.compile(r"^[^a-zA-Z0-9]+$")

def rule_advertisement(text: str) -> int:
    return int(bool(URL.search(text) or COUPON.search(text) or PHONE.search(text)))

def rule_rant_no_visit(text: str) -> int:
    return int(bool(NO_VISIT.search(text)))

def rule_spam(text: str) -> int:
    return int(bool(URL.search(text) and CONTACT_ME.search(text)) or bool(GIBBERISH.match(text)))

def rule_low_quality(text: str) -> int:
    return int(len(text.split()) < 3)

def rule_irrelevant_from_relevancy(relevancy_score: float, tau: float = 0.25) -> int:
    return int(relevancy_score < tau)
