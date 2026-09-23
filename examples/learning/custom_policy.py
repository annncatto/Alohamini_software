"""A minimal custom execution policy built on a native checkpoint.

This example chooses receding-horizon execution (only action 0 of each chunk),
without implementing another image decoder, normalizer or robot control loop.
It is a Python policy interface, not a requirement to inherit a platform class.
Set ALOHAMINI_CHECKPOINT to a trained native checkpoint before selecting this factory.
"""

import os

from alohamini.learning.policy import NativePolicy


class RecedingHorizonPolicy:
    def __init__(self, checkpoint, device="cuda"):
        self.adapter = NativePolicy(
            checkpoint, device=device, n_action_steps=1, temporal_ensemble_coeff=None
        )
        self.robot_metadata = self.adapter.robot_metadata

    def reset(self):
        """Called at each episode start; discard every queued/ensembled old action."""
        self.adapter.reset()

    def select_action(self, snapshot):
        """Input: HostSnapshot; output: all named absolute Host targets, never motor writes.

        To develop a different network, replace adapter.model.select_action with
        your tensor policy: normalized BCHW RGB/optional BxS state -> Bx18 action.
        Training, stats, feature order and output units must agree with the checkpoint.
        The evaluator retains calibration, target-range, watchdog and ownership checks.
        """
        return self.adapter.select_action(snapshot)


def create():
    return RecedingHorizonPolicy(os.environ["ALOHAMINI_CHECKPOINT"])
