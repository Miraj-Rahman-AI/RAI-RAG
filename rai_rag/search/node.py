from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SearchState:
    """
    State that SI-MCTS searches over.
    """
    user_prompt: str
    ir: Dict[str, Any]
    plan: Dict[str, Any]          # retrieval plan
    evidence: Dict[str, Any]      # evidence bundle
    draft_answer: str = ""        # candidate answer (optional)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    """
    MCTS node for (state + action).
    """
    state: SearchState
    parent: Optional["Node"] = None
    children: Dict[str, "Node"] = field(default_factory=dict)

    # MCTS stats
    N: int = 0
    W: float = 0.0
    Q: float = 0.0

    # action taken from parent to reach this node
    action: Optional[str] = None

    # terminal marker
    terminal: bool = False

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_child(self, action: str, child_state: SearchState) -> "Node":
        child = Node(state=child_state, parent=self, action=action)
        self.children[action] = child
        return child

    def update(self, value: float) -> None:
        self.N += 1
        self.W += float(value)
        self.Q = self.W / max(1, self.N)

