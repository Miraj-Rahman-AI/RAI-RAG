import pytest


@pytest.mark.skip(reason="SI-MCTS core not implemented yet. Enable once rai_rag.search.simcts exists.")
def test_simcts_placeholder():
    """
    Once SI-MCTS is implemented, replace this with real tests that check:
    - nodes represent (trace, retrieval_action, evidence_state)
    - safety-dominant pruning removes unsafe branches early
    - backprop favors safe trajectories
    - cache avoids repeated judge calls
    """
    assert True

