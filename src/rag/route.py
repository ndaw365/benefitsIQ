"""Route a question to the policy documents it names, so retrieval searches only those."""

import re

# Names each document uses for itself (insurer, policyholder, title). In production this
# would come from the policy admin system, not a hand-written table.
POLICY_ALIASES: dict[str, tuple[str, ...]] = {
    "lincoln_moravian_ltd": ("lincoln", "moravian"),
    "standard_uah_ltd": ("standard insurance", "uah", "huntsville", "university of alabama"),
    "securian_colorado_term_life": ("securian", "minnesota life", "colorado"),
    "dol_fmla_employee_guide": ("fmla", "family and medical leave act"),
}


def detect_policies(question: str) -> list[str]:
    """Return the doc_ids whose aliases appear as whole words in the question; [] means 'search all'."""
    text = question.lower()
    return [
        doc_id
        for doc_id, aliases in POLICY_ALIASES.items()
        if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases)
    ]


def strip_policy_names(question: str) -> str:
    """Remove policy aliases from the question once they've been used for routing.

    Inside the right document, a name like "Lincoln" still pulls the query embedding toward
    the cover page, which is the only page that says "Lincoln". Falls back to the original
    question if nothing would be left.
    """
    text = question
    for aliases in POLICY_ALIASES.values():
        for alias in aliases:
            text = re.sub(rf"\b{re.escape(alias)}\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text if re.search(r"[a-z]{3,}", text, re.IGNORECASE) else question
