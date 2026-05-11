"""
Variance-reduction IS gradient estimators — PPO (clipped) vs PPO-NoClip (unclipped).

Both groups use GAE (gae_lambda=0.95). The independent variable is the IS weight:

  standard   : ρ = π_new/π_old  (baseline)
  snis       : ρ / mean(ρ)  — Self-Normalized IS
  defensive  : ρ / (α + (1-α)ρ),  α=0.1  — smooth mixture bound
  power      : ρ^β,  β=0.5  — Rényi-2 optimal
  truncated  : min(ρ, c),  c=2  — hard variance bound

PPO uses inside placement: L = min( w·A,  clip(w, 1-ε, 1+ε)·A )
PPO-NoClip: unclipped  w·A
"""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


# ---------------------------------------------------------------------------
# Shared logging helper
# ---------------------------------------------------------------------------
def _log_ppo_stats(model, entropy_losses, pg_losses, value_losses,
                   clip_fracs, approx_kl_divs, loss, extra=None):
    clip_range = model.clip_range(model._current_progress_remaining)
    ev = explained_variance(
        model.rollout_buffer.values.flatten(),
        model.rollout_buffer.returns.flatten(),
    )
    model.logger.record("train/entropy_loss",         np.mean(entropy_losses))
    model.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    model.logger.record("train/value_loss",           np.mean(value_losses))
    model.logger.record("train/approx_kl",            np.mean(approx_kl_divs))
    model.logger.record("train/clip_fraction",        np.mean(clip_fracs))
    model.logger.record("train/loss",                 loss.item())
    model.logger.record("train/explained_variance",   ev)
    model.logger.record("train/n_updates",  model._n_updates, exclude="tensorboard")
    model.logger.record("train/clip_range", clip_range)
    if hasattr(model.policy, "log_std"):
        model.logger.record("train/std", th.exp(model.policy.log_std).mean().item())
    for k, v in (extra or {}).items():
        model.logger.record(k, v)


# ---------------------------------------------------------------------------
# Shared PPO train loop — one weight_fn, one placement flag
# ---------------------------------------------------------------------------
def _ppo_train(model, weight_fn) -> None:
    """PPO update: L = min(w·A, clip(w, 1-ε, 1+ε)·A)."""
    model.policy.set_training_mode(True)
    model._update_learning_rate(model.policy.optimizer)
    clip_range    = model.clip_range(model._current_progress_remaining)
    clip_range_vf = (model.clip_range_vf(model._current_progress_remaining)
                     if model.clip_range_vf is not None else None)

    entropy_losses, pg_losses, value_losses, clip_fracs = [], [], [], []
    approx_kl_divs, ratio_means = [], []
    continue_training = True
    loss = th.tensor(0.0)

    for epoch in range(model.n_epochs):
        for rollout_data in model.rollout_buffer.get(model.batch_size):
            actions = rollout_data.actions
            if isinstance(model.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            values, log_prob, entropy = model.policy.evaluate_actions(
                rollout_data.observations, actions)
            values = values.flatten()

            advantages = rollout_data.advantages
            if model.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = th.exp(log_prob - rollout_data.old_log_prob)
            w = weight_fn(ratio)
            ratio_means.append(w.detach().mean().item())

            pg1 = advantages * w
            pg2 = advantages * th.clamp(w, 1 - clip_range, 1 + clip_range)
            clip_fracs.append(
                th.mean((th.abs(w.detach() - 1) > clip_range).float()).item())

            policy_loss = -th.min(pg1, pg2).mean()
            pg_losses.append(policy_loss.item())

            if clip_range_vf is None:
                values_pred = values
            else:
                values_pred = rollout_data.old_values + th.clamp(
                    values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses.append(value_loss.item())

            entropy_loss = (-th.mean(entropy) if entropy is not None
                            else -th.mean(-log_prob))
            entropy_losses.append(entropy_loss.item())

            loss = policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss

            with th.no_grad():
                log_ratio = log_prob - rollout_data.old_log_prob
                approx_kl = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                approx_kl_divs.append(approx_kl)

            if model.target_kl is not None and approx_kl > 1.5 * model.target_kl:
                continue_training = False
                break

            model.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()

        model._n_updates += 1
        if not continue_training:
            break

    _log_ppo_stats(model, entropy_losses, pg_losses, value_losses,
                   clip_fracs, approx_kl_divs, loss,
                   extra={"train/ratio_mean": np.mean(ratio_means) if ratio_means else float("nan")})


# ---------------------------------------------------------------------------
# PPO estimator classes — placement attribute set by make_model
# ---------------------------------------------------------------------------
class SNISPPO(PPO):
    """PPO + SNIS: w = ρ / mean(ρ)."""
    def train(self) -> None:
        _ppo_train(self, lambda r: r / r.detach().mean().clamp(min=1e-8))


class DefensivePPO(PPO):
    """PPO + Defensive IS: w = ρ / (α + (1-α)ρ), α=0.1."""
    ALPHA: float = 0.1

    def train(self) -> None:
        a = self.ALPHA
        _ppo_train(self, lambda r: r / (a + (1 - a) * r))


class PowerPPO(PPO):
    """PPO + Power IS: w = ρ^β, β=0.5."""
    BETA: float = 0.5

    def train(self) -> None:
        b = self.BETA
        _ppo_train(self, lambda r: r.pow(b))


class TruncatedPPO(PPO):
    """PPO + Truncated IS: w = min(ρ, c), c=2."""
    CAP: float = 2.0

    def train(self) -> None:
        c = self.CAP
        _ppo_train(self, lambda r: th.clamp(r, max=c))


# ---------------------------------------------------------------------------
# Shared VPG train loop (PPO-NoClip) — no placement, just w·A
# ---------------------------------------------------------------------------
def _vpg_train(model: PPO, weight_fn) -> None:
    """PPO-NoClip: unclipped objective  -mean(weight_fn(ρ)·A).
    Log-ratio clamped to [-5, 5] to prevent explosion across epochs.
    """
    model.policy.set_training_mode(True)
    model._update_learning_rate(model.policy.optimizer)
    clip_range_vf = (model.clip_range_vf(model._current_progress_remaining)
                     if model.clip_range_vf is not None else None)

    entropy_losses, pg_losses, value_losses = [], [], []
    approx_kl_divs, w_means = [], []
    loss = th.tensor(0.0)

    for epoch in range(model.n_epochs):
        for rollout_data in model.rollout_buffer.get(model.batch_size):
            actions = rollout_data.actions
            if isinstance(model.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            values, log_prob, entropy = model.policy.evaluate_actions(
                rollout_data.observations, actions)
            values = values.flatten()

            advantages = rollout_data.advantages
            if model.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            log_diff = th.clamp(log_prob - rollout_data.old_log_prob, -5.0, 5.0)
            ratio = th.exp(log_diff)
            w = weight_fn(ratio)
            w_means.append(w.detach().mean().item())

            policy_loss = -(w * advantages).mean()
            pg_losses.append(policy_loss.item())

            if clip_range_vf is None:
                values_pred = values
            else:
                values_pred = rollout_data.old_values + th.clamp(
                    values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            value_losses.append(value_loss.item())

            entropy_loss = (-th.mean(entropy) if entropy is not None
                            else -th.mean(-log_prob))
            entropy_losses.append(entropy_loss.item())

            loss = policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss

            if not th.isfinite(loss):
                continue

            with th.no_grad():
                log_ratio = log_prob - rollout_data.old_log_prob
                approx_kl = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                approx_kl_divs.append(approx_kl)

            model.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), model.max_grad_norm)
            model.policy.optimizer.step()

        model._n_updates += 1

    _log_ppo_stats(model, entropy_losses, pg_losses, value_losses,
                   [], approx_kl_divs, loss,
                   extra={"train/ratio_mean": np.mean(w_means) if w_means else float("nan")})


# ---------------------------------------------------------------------------
# PPO-NoClip classes
# ---------------------------------------------------------------------------
class VanillaPG(PPO):
    """PPO-NoClip + Standard IS. loss = -mean(ρ·A)."""
    def train(self) -> None:
        _vpg_train(self, lambda r: r)


class VanillaPGSNIS(PPO):
    """PPO-NoClip + SNIS: w = ρ / mean(ρ)."""
    def train(self) -> None:
        _vpg_train(self, lambda r: r / r.detach().mean().clamp(min=1e-8))


class VanillaPGDefensive(PPO):
    """PPO-NoClip + Defensive IS: w = ρ / (α + (1-α)ρ), α=0.1."""
    ALPHA: float = 0.1
    def train(self) -> None:
        a = self.ALPHA
        _vpg_train(self, lambda r: r / (a + (1 - a) * r))


class VanillaPGPower(PPO):
    """PPO-NoClip + Power IS: w = ρ^β, β=0.5."""
    BETA: float = 0.5
    def train(self) -> None:
        b = self.BETA
        _vpg_train(self, lambda r: r.pow(b))


class VanillaPGTruncated(PPO):
    """PPO-NoClip + Truncated IS: w = min(ρ, c), c=2."""
    CAP: float = 2.0
    def train(self) -> None:
        c = self.CAP
        _vpg_train(self, lambda r: th.clamp(r, max=c))


# ---------------------------------------------------------------------------
# Mode lists for runner
# ---------------------------------------------------------------------------
PPO_MODES = ["standard", "snis", "defensive", "power", "truncated"]
REINFORCE_MODES = ["standard", "snis", "defensive", "power", "truncated"]
