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
