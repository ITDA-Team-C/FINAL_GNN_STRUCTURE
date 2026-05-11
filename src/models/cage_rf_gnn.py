"""
CAGE-RF GNN - Backward Compatibility Wrapper
기존 코드와의 호환성을 위해 cage_rf_gnn_sage로 리다이렉트합니다.
새로운 코드에서는 cage_rf_gnn_sage를 직접 사용하세요.
"""

from src.models.cage_rf_gnn_sage import CAGERF_GNN, CAGERFGNNBranch, RelationGate

__all__ = ["CAGERF_GNN", "CAGERFGNNBranch", "RelationGate"]
