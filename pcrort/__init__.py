"""PC-RORT: provenance-constrained risk-optimal relation transport."""
from .matrix import conditional_risk_matrix, relative_contrast, two_candidate_robust_selection
from .evidence import EvidenceContract, EvidenceUnit, DecisionRecord

__all__ = ["conditional_risk_matrix", "relative_contrast",
           "two_candidate_robust_selection", "EvidenceContract",
           "EvidenceUnit", "DecisionRecord"]
