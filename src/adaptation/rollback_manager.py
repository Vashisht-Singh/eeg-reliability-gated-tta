"""
Rollback and Model Recovery Manager.
Maintains stable checkpoints and restores model weights when online divergence is detected.
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple


class RollbackManager:
    """
    Monitors online adaptation stability and restores stable checkpoints
    without accessing target domain labels.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_interval: int = 50,
        max_consecutive_rejections: int = 25,
        max_entropy_explosion: float = 0.69
    ):
        self.source_state = copy.deepcopy(model.state_dict())
        self.last_stable_state = copy.deepcopy(model.state_dict())
        self.checkpoint_interval = checkpoint_interval
        self.max_consecutive_rejections = max_consecutive_rejections
        self.max_entropy_explosion = max_entropy_explosion

        self.consecutive_rejections = 0
        self.accepted_updates_since_checkpoint = 0
        self.total_rollbacks = 0
        self.rollback_history = []

    def register_step(
        self,
        model: nn.Module,
        was_accepted: bool,
        signals: Dict[str, float],
        timestamp: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Record step outcome and determine if rollback is triggered.
        Returns: (needs_rollback, reason)
        """
        if was_accepted:
            self.consecutive_rejections = 0
            self.accepted_updates_since_checkpoint += 1

            # Update stable checkpoint after sustained successful adaptation
            if self.accepted_updates_since_checkpoint >= self.checkpoint_interval:
                self.last_stable_state = copy.deepcopy(model.state_dict())
                self.accepted_updates_since_checkpoint = 0
        else:
            self.consecutive_rejections += 1

        # Check Instability Condition 1: Consecutive rejections threshold exceeded
        if self.consecutive_rejections >= self.max_consecutive_rejections:
            self.total_rollbacks += 1
            reason = f"consecutive_rejections_exceeded_{self.consecutive_rejections}"
            self.rollback_history.append({'timestamp': timestamp, 'reason': reason, 'type': 'to_last_stable'})
            self.consecutive_rejections = 0
            return True, reason

        # Check Instability Condition 2: Entropy explosion near maximum uniform entropy (0.693)
        ent = signals.get('candidate_entropy', 0.0)
        if ent >= self.max_entropy_explosion:
            self.total_rollbacks += 1
            reason = f"entropy_explosion_{ent:.4f}_ge_{self.max_entropy_explosion}"
            self.rollback_history.append({'timestamp': timestamp, 'reason': reason, 'type': 'to_source'})
            return True, reason

        return False, None

    def execute_rollback(self, model: nn.Module, to_source: bool = False):
        """Restore model weights to stable or source checkpoint."""
        target_state = self.source_state if to_source else self.last_stable_state
        model.load_state_dict(copy.deepcopy(target_state))
