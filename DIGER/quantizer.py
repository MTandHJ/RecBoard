import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import center_distance_for_constraint, sinkhorn_algorithm

__all__ = [
    "AutoSigmaGaussian",
    "AutoSigmaGumbel",
    "AutoSigmaSimple",
    "ResidualQuantizerOutput",
    "ResidualQuantizer",
    "VectorQuantizer",
]


@dataclass
class ResidualQuantizerOutput:
    quantized: torch.Tensor
    loss: torch.Tensor
    ids: torch.Tensor
    one_hots: torch.Tensor
    distances: torch.Tensor
    sigma: Optional[torch.Tensor] = None


class AutoSigmaGaussian(nn.Module):
    r"""Gaussian-noise relaxation with a learnable uncertainty scale."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(self._initial_sigma(cfg)))

    @staticmethod
    def _initial_sigma(cfg) -> float:
        if cfg.initial_std is not None:
            initial_std = float(cfg.initial_std)
            return -20.0 if initial_std <= 1.0e-5 else math.log2(initial_std)
        return float(cfg.initial_sigma)

    def forward(
        self,
        logits: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
        dim: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.pow(2.0, self.sigma).clamp(min=1.0e-5, max=100.0)
        noisy_logits = logits + torch.randn_like(logits) * scale if self.training else logits
        probs = (noisy_logits / tau).softmax(dim)
        if not hard:
            return probs, self.sigma

        index = probs.max(dim, keepdim=True)[1]
        hard_probs = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return hard_probs - probs.detach() + probs, self.sigma


class AutoSigmaGumbel(nn.Module):
    r"""Gumbel relaxation with a learnable uncertainty scale."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(AutoSigmaGaussian._initial_sigma(cfg)))

    def forward(
        self,
        logits: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
        dim: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            scale = torch.pow(2.0, self.sigma).clamp(min=1.0e-5, max=100.0)
            noise = -torch.empty_like(logits).exponential_().log() * scale
            probs = ((logits + noise) / tau).softmax(dim)
        else:
            probs = (logits / tau).softmax(dim)
        if not hard:
            return probs, self.sigma

        index = probs.max(dim, keepdim=True)[1]
        hard_probs = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return hard_probs - probs.detach() + probs, self.sigma

    @staticmethod
    def compute_uncertainty_loss(
        task_loss: torch.Tensor,
        sigma: torch.Tensor,
        reg_weight: float = 1.0,
    ) -> torch.Tensor:
        if task_loss.detach() >= 2.0:
            k, c = 0.458145, 1.361442
        else:
            k, c = 0.018127, 0.036916
        return task_loss * torch.exp(-k * sigma) + c * reg_weight * sigma


class AutoSigmaSimple(nn.Module):
    r"""Simple uncertainty relaxation used by the SDUD variant."""

    def __init__(self, cfg) -> None:
        super().__init__()
        initial_std = (
            float(cfg.initial_std) if cfg.initial_std is not None else float(cfg.initial_sigma)
        )
        self.sigma = nn.Parameter(torch.tensor(initial_std))

    def forward(
        self,
        logits: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
        dim: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.sigma.abs().clamp(min=1.0e-5, max=100.0)
        if self.training:
            noise = -torch.empty_like(logits).exponential_().log() * scale
            probs = ((logits + noise) / tau).softmax(dim)
        else:
            probs = (logits / tau).softmax(dim)
        if not hard:
            return probs, self.sigma

        index = probs.max(dim, keepdim=True)[1]
        hard_probs = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return hard_probs - probs.detach() + probs, self.sigma

    def compute_uncertainty_loss(
        self,
        task_loss: torch.Tensor,
        sigma: torch.Tensor,
        lambda_bias: float = 0.5,
    ) -> torch.Tensor:
        effective_sigma = (sigma.abs() + lambda_bias).clamp(min=1.0e-6)
        return task_loss / (2 * effective_sigma.pow(2)) + torch.log(effective_sigma)


class VectorQuantizer(nn.Module):
    r"""Single residual quantization layer with optional DIGER relaxations."""

    def __init__(self, cfg, num_codewords: int, sk_epsilon: float = 0.0) -> None:
        super().__init__()

        self.num_codewords = num_codewords
        self.codebook_dim = int(cfg.codebook_dim)
        self.commit_weight = float(cfg.commit_weight)
        self.dist = cfg.dist
        self.sk_epsilon = float(sk_epsilon)
        self.sk_iters = int(cfg.sk_iters)
        self.gumbel_hard_switch_epoch = int(cfg.gumbel_hard_switch_epoch)

        self.codebook = nn.Embedding(num_codewords, self.codebook_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codewords, 1.0 / num_codewords)

        self.use_adaptive_selection = bool(cfg.use_adaptive_selection)
        self.usage_momentum = float(cfg.usage_momentum)
        self.hot_threshold_ratio = float(cfg.hot_threshold_ratio)
        self.register_buffer("code_usage_ema", torch.ones(num_codewords) / num_codewords)
        self.reset_adaptive_selection_stats()

        self.use_learnable_sigma = bool(cfg.use_learnable_sigma_gumbel)
        if self.use_learnable_sigma:
            if cfg.noise_type == "gaussian":
                self.auto_sigma = AutoSigmaGaussian(cfg)
            elif bool(cfg.use_simple_uncertainty_loss):
                self.auto_sigma = AutoSigmaSimple(cfg)
            else:
                self.auto_sigma = AutoSigmaGumbel(cfg)

    def reset_adaptive_selection_stats(self) -> None:
        self._sampled_count = 0
        self._deterministic_count = 0

    def get_adaptive_selection_stats(self) -> dict:
        total = self._sampled_count + self._deterministic_count
        return {
            "sampled_count": self._sampled_count,
            "deterministic_count": self._deterministic_count,
            "total_count": total,
            "sampled_ratio": self._sampled_count / total if total else 0.0,
            "deterministic_ratio": self._deterministic_count / total if total else 0.0,
        }

    def get_codebook(self) -> torch.Tensor:
        return self.codebook.weight

    def pairwise_distance(self, x: torch.Tensor) -> torch.Tensor:
        if self.dist == "l2":
            return (
                x.pow(2).sum(dim=1, keepdim=True)
                + self.codebook.weight.pow(2).sum(dim=1).unsqueeze(0)
                - 2 * x @ self.codebook.weight.t()
            )
        if self.dist == "dot":
            return -(x @ self.codebook.weight.t())
        if self.dist == "cos":
            return -(F.normalize(x, dim=-1) @ F.normalize(self.codebook.weight, dim=-1).t())
        raise NotImplementedError(f"unexpected distance {self.dist!r}")

    @torch.no_grad()
    def get_indices(self, x: torch.Tensor, use_sinkhorn: bool = True) -> torch.Tensor:
        distances = self.pairwise_distance(x.view(-1, self.codebook_dim))
        if use_sinkhorn and self.sk_epsilon > 0:
            centered_distances = center_distance_for_constraint(distances).double()
            assignments = sinkhorn_algorithm(centered_distances, self.sk_epsilon, self.sk_iters)
            return assignments.argmax(dim=-1).view(x.shape[:-1])
        return distances.argmin(dim=-1).view(x.shape[:-1])

    def sample_indices(
        self,
        logits: torch.Tensor,
        deterministic_ids: torch.Tensor,
        tau: float,
        use_sampling: bool,
        current_epoch: Optional[int],
        sample_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        use_hard = (
            current_epoch is not None
            and self.gumbel_hard_switch_epoch > 0
            and current_epoch >= self.gumbel_hard_switch_epoch
        )
        if hasattr(self, "auto_sigma"):
            probs, sigma = self.auto_sigma(logits, tau=tau, hard=use_hard, dim=-1)
        else:
            probs = F.gumbel_softmax(logits, tau=tau, hard=use_hard, dim=-1)
            sigma = None

        if not use_sampling:
            return deterministic_ids, (logits / tau).softmax(dim=-1), sigma

        sampled_ids = probs.argmax(dim=-1)
        if not self.use_adaptive_selection:
            self._sampled_count += int(sampled_ids.numel())
            return sampled_ids, probs, sigma

        with torch.no_grad():
            counts = torch.bincount(deterministic_ids, minlength=self.num_codewords).float()
            freqs = counts / counts.sum().clamp_min(1)
            self.code_usage_ema.mul_(self.usage_momentum).add_(
                freqs,
                alpha=1.0 - self.usage_momentum,
            )
        if sample_mask is None:
            hot_threshold = self.hot_threshold_ratio / self.num_codewords
            use_sampled = self.code_usage_ema[deterministic_ids] > hot_threshold
        else:
            use_sampled = sample_mask.view_as(deterministic_ids)
        deterministic_probs = (logits / tau).softmax(dim=-1)
        selected_ids = torch.where(use_sampled, sampled_ids, deterministic_ids)
        probs = torch.where(use_sampled[:, None], probs, deterministic_probs)

        self._sampled_count += int(use_sampled.sum().item())
        self._deterministic_count += int((~use_sampled).sum().item())
        return selected_ids, probs, sigma

    @torch.no_grad()
    def get_hot_code_mask(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        r"""Identify samples assigned to frequently used codewords.

        Parameters
        ----------
        x : torch.Tensor
            Latent vectors assigned by the first residual quantizer.

        Returns
        -------
        Optional[torch.Tensor]
            Boolean mask with one value per sample. ``True`` means the
            deterministic assignment lands on a hot codeword, so all residual
            quantizer layers should use the sampled Gumbel assignment for that
            sample. ``None`` is returned when adaptive selection is disabled.
        """
        if not self.use_adaptive_selection:
            return None

        latents = x.view(-1, self.codebook_dim)
        distances = self.pairwise_distance(latents)
        deterministic_ids = distances.argmin(dim=-1)
        hot_threshold = self.hot_threshold_ratio / self.num_codewords
        return self.code_usage_ema[deterministic_ids] > hot_threshold

    def forward(
        self,
        x: torch.Tensor,
        use_sinkhorn: Optional[bool] = None,
        use_gumbel: bool = False,
        tau: float = 1.0,
        use_indicator_ste: bool = True,
        use_sampling: bool = True,
        current_epoch: Optional[int] = None,
        sample_mask: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        r"""Quantize one residual latent tensor with one codebook.

        Parameters
        ----------
        x : torch.Tensor
            Residual latent tensor whose last dimension equals ``codebook_dim``.
        use_sinkhorn : bool, optional
            Whether to use balanced Sinkhorn assignment for deterministic IDs. If
            ``None``, Sinkhorn is enabled when this layer has positive epsilon.
        use_gumbel : bool, default=False
            Whether training uses Gumbel-based sampled assignments.
        tau : float, default=1.0
            Softmax temperature for Gumbel or deterministic probabilities.
        use_indicator_ste : bool, default=True
            Whether to apply the straight-through estimator on one-hot indicators
            instead of quantized embedding vectors.
        use_sampling : bool, default=True
            Whether sampled IDs may replace deterministic IDs in the Gumbel path.
        current_epoch : int, optional
            Current training epoch used for hard Gumbel switching.
        sample_mask : torch.Tensor, optional
            Shared hot-code mask from the first residual quantizer. ``True`` means
            using sampled IDs for the corresponding sample.

        Returns
        -------
        tuple
            Quantized residuals, VQ loss, selected IDs, deterministic one-hot
            assignments, pairwise distances, and optional sigma.
        """
        latents = x.view(-1, self.codebook_dim)
        distances = self.pairwise_distance(latents)
        deterministic_ids = distances.argmin(dim=-1)
        logits_distances = distances

        if use_sinkhorn is None:
            use_sinkhorn = self.sk_epsilon > 0
        if use_sinkhorn and self.sk_epsilon > 0:
            centered_distances = center_distance_for_constraint(distances).double()
            assignments = sinkhorn_algorithm(centered_distances, self.sk_epsilon, self.sk_iters)
            deterministic_ids = assignments.argmax(dim=-1)
            logits_distances = centered_distances.float()

        one_hot = F.one_hot(deterministic_ids, self.num_codewords).float()
        sigma = None
        if self.training and use_gumbel:
            selected_ids, probs, sigma = self.sample_indices(
                -logits_distances.float(),
                deterministic_ids,
                tau=tau,
                use_sampling=use_sampling,
                current_epoch=current_epoch,
                sample_mask=sample_mask,
            )
            hard_probs = F.one_hot(selected_ids, self.num_codewords).float()
            if use_indicator_ste:
                q = (hard_probs - probs.detach() + probs) @ self.codebook.weight
            else:
                soft_q = probs @ self.codebook.weight
                hard_q = self.codebook(selected_ids)
                q = hard_q + (soft_q - soft_q.detach())
            ids = selected_ids
        else:
            ids = deterministic_ids
            q = self.codebook(ids)
            q = latents + (q - latents).detach()

        q = q.view_as(x)
        if self.dist == "l2":
            codebook_loss = F.mse_loss(q, x.detach())
            commitment_loss = F.mse_loss(q.detach(), x)
            loss = codebook_loss + self.commit_weight * commitment_loss
        else:
            loss = self.commit_weight * F.cross_entropy(-distances, ids.detach())

        return q, loss, ids.view(x.shape[:-1]), one_hot, distances, sigma


class ResidualQuantizer(nn.Module):
    r"""Residual stack of DIGER vector quantizers."""

    def __init__(self, cfg) -> None:
        super().__init__()

        num_codewords = cfg.num_codewords
        if isinstance(num_codewords, int):
            num_codewords = [num_codewords] * int(cfg.num_codebooks)
        self.num_codewords = list(num_codewords)

        sk_epsilons = list(cfg.sk_epsilons)
        if len(sk_epsilons) != len(self.num_codewords):
            sk_epsilons = [sk_epsilons[0] if sk_epsilons else 0.0] * len(self.num_codewords)

        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(cfg, num_codewords=count, sk_epsilon=epsilon)
                for count, epsilon in zip(self.num_codewords, sk_epsilons)
            ]
        )

    def reset_adaptive_selection_stats(self) -> None:
        for quantizer in self.quantizers:
            quantizer.reset_adaptive_selection_stats()

    def get_adaptive_selection_stats(self) -> dict:
        per_layer = [quantizer.get_adaptive_selection_stats() for quantizer in self.quantizers]
        sampled = sum(stats["sampled_count"] for stats in per_layer)
        deterministic = sum(stats["deterministic_count"] for stats in per_layer)
        total = sampled + deterministic
        return {
            "sampled_count": sampled,
            "deterministic_count": deterministic,
            "total_count": total,
            "sampled_ratio": sampled / total if total else 0.0,
            "deterministic_ratio": deterministic / total if total else 0.0,
            "per_layer": per_layer,
        }

    def compute_simple_uncertainty_loss(
        self,
        task_loss: torch.Tensor,
        sigma: torch.Tensor,
        lambda_bias: float,
    ) -> torch.Tensor:
        auto_sigma = getattr(self.quantizers[0], "auto_sigma", None)
        if not isinstance(auto_sigma, AutoSigmaSimple):
            return task_loss
        return auto_sigma.compute_uncertainty_loss(task_loss, sigma, lambda_bias=lambda_bias)

    @torch.no_grad()
    def get_indices(self, x: torch.Tensor, use_sinkhorn: bool = True) -> torch.Tensor:
        residual = x
        ids = []
        for quantizer in self.quantizers:
            q, _, level_ids, _, _, _ = quantizer(residual, use_sinkhorn=use_sinkhorn)
            residual = residual - q
            ids.append(level_ids)
        return torch.stack(ids, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        use_gumbel: bool = False,
        tau: float = 1.0,
        use_indicator_ste: bool = True,
        use_sampling: bool = True,
        current_epoch: Optional[int] = None,
    ) -> ResidualQuantizerOutput:
        r"""Quantize latents through the residual codebook stack.

        Parameters
        ----------
        x : torch.Tensor
            Latent tensor emitted by the ID encoder.
        use_gumbel : bool, default=False
            Whether each quantizer may use sampled Gumbel assignments in training.
        tau : float, default=1.0
            Softmax temperature for Gumbel relaxation.
        use_indicator_ste : bool, default=True
            Whether each layer applies STE on one-hot indicators.
        use_sampling : bool, default=True
            Whether sampled IDs are allowed before ``stop_gumbel_sampling_epoch``.
        current_epoch : int, optional
            Current training epoch used by Gumbel hard switching.

        Returns
        -------
        ResidualQuantizerOutput
            Aggregated quantized latent, mean VQ loss, stacked code IDs, stacked
            one-hot assignments, stacked distances, and optional mean sigma.
        """
        residual = x
        q = torch.zeros_like(x)
        losses, ids, one_hots, distances, sigmas = [], [], [], [], []
        hot_code_mask = None
        if use_gumbel and use_sampling:
            hot_code_mask = self.quantizers[0].get_hot_code_mask(x)

        for quantizer in self.quantizers:
            level_q, loss, level_ids, one_hot, distance, sigma = quantizer(
                residual,
                use_gumbel=use_gumbel,
                tau=tau,
                use_indicator_ste=use_indicator_ste,
                use_sampling=use_sampling,
                current_epoch=current_epoch,
                sample_mask=hot_code_mask,
            )
            residual = residual - level_q
            q = q + level_q
            losses.append(loss)
            ids.append(level_ids)
            one_hots.append(one_hot)
            distances.append(distance)
            if sigma is not None:
                sigmas.append(sigma)

        return ResidualQuantizerOutput(
            quantized=q,
            loss=torch.stack(losses).mean(),
            ids=torch.stack(ids, dim=-1),
            one_hots=torch.stack(one_hots, dim=1),
            distances=torch.stack(distances, dim=1),
            sigma=torch.stack(sigmas).mean() if sigmas else None,
        )
