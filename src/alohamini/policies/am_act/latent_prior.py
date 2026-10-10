"""Diagonal visual Gaussian prior and analytic conditional KL."""

from torch import nn


class ConditionalGaussianPrior(nn.Module):
    def __init__(self, dim, queries, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * queries, dim), nn.GELU(), nn.Linear(dim, latent_dim * 2)
        )

    def forward(self, visual_tokens):
        mean, logvar = self.net(visual_tokens.flatten(1)).chunk(2, dim=-1)
        return mean, logvar.clamp(-20, 10)


def gaussian_kl(mean_q, logvar_q, mean_p, logvar_p):
    """Per-dimension KL(q || p), evaluated in float32 under mixed precision."""
    mean_q, logvar_q, mean_p, logvar_p = [x.float() for x in (mean_q, logvar_q, mean_p, logvar_p)]
    return 0.5 * (
        logvar_p
        - logvar_q
        + (logvar_q - logvar_p).exp()
        + (mean_q - mean_p).square() * (-logvar_p).exp()
        - 1
    )
