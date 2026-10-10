"""Label physical velocities before normalization, using the frequency-fit rule."""

import torch

from alohamini.learning.processor import Processor

from .classification import nearest_class


class AMACTProcessor(Processor):
    @classmethod
    def from_config(cls, config, stats, device="cpu"):
        processor = super().from_config(config, stats, device)
        processor.discrete_dims = config.discrete_action_dims
        processor.discrete_values = config.discrete_action_values
        return processor

    def __call__(self, batch):
        if self.discrete_dims and "action" in batch:
            # Numeric targets only. No extra image conversion or video decoding.
            batch = {
                **batch,
                "discrete_action_labels": torch.stack(
                    [
                        nearest_class(batch["action"][..., dim], values)
                        for dim, values in zip(
                            self.discrete_dims, self.discrete_values, strict=True
                        )
                    ],
                    dim=-1,
                ),
            }
        return super().__call__(batch)
