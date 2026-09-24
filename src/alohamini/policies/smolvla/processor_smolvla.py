"""SmolVLA's official task/tokenization order with shared numeric transforms."""

import torch

from alohamini.learning.processor import Processor


def fit_statistics(samples):
    """Fit training numeric fields; RGB is IDENTITY before SigLIP conversion."""
    keys = [k for k in samples.sample_keys if not k.startswith("observation.images.")]
    stats = samples.statistics(keys=keys)
    for key in samples.input_features:
        if key.startswith("observation.images."):
            stats[key] = {"mean": [[[0.0]]] * 3, "std": [[[1.0]]] * 3}
    return stats


class SmolVLAProcessor:
    def __init__(self, config, stats, tokenizer, device="cpu"):
        self.config, self.tokenizer = config, tokenizer
        self.numeric = Processor.from_config(config, stats, device)

    def __call__(self, batch):
        tasks = batch.get("task")
        if isinstance(tasks, str):
            tasks = [tasks]
        if (
            not isinstance(tasks, (list, tuple))
            or not tasks
            or any(not isinstance(t, str) or not t.strip() for t in tasks)
        ):
            raise ValueError("SmolVLA requires one nonempty task description per sample")
        if len(tasks) != batch["observation.state"].shape[0]:
            raise ValueError("Task count does not match the observation batch")
        tokenized = self.tokenizer(
            [t if t.endswith("\n") else t + "\n" for t in tasks],
            max_length=self.config.tokenizer_max_length,
            truncation=True,
            padding=self.config.pad_language_to,
            padding_side="right",
            return_tensors="pt",
        )
        tensors = {k: v for k, v in batch.items() if k != "task"}
        tensors["observation.language.tokens"] = tokenized["input_ids"]
        tensors["observation.language.attention_mask"] = tokenized["attention_mask"].bool()
        for key in self.config.image_features:
            if key not in tensors and ".empty_camera_" not in key:
                raise ValueError(f"Missing configured camera: {key}")
            if key in tensors:
                image = tensors[key]
                if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise ValueError(f"{key}: expected finite RGB in [0, 1]")
        return self.numeric(tensors)

    def action(self, tensor):
        return self.numeric.action(tensor)
