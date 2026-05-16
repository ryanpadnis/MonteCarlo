"""
REINFORCE with trajectory reuse + Importance Sampling.

The core experiment:
  1. Collect N steps with the CURRENT policy (frozen as π_old).
  2. Perform K gradient epochs on that SAME batch.
  3. Without IS correction, reusing stale data biases the PG gradient.
     IS weights  ρ = π_θ(a|s) / π_old(a|s)  correct for this — at the
     cost of added variance.

Modes
-----
  standard   — 1 epoch, w=1     on-policy REINFORCE  (gold standard)
  naive      — K epochs, w=1    biased reuse → should diverge
  is         — K epochs, surrogate  ρ·A         (unbiased, high variance)
  snis       — K epochs, surrogate  (ρ / mean ρ)·A   (self-normalized)
  truncated  — K epochs, surrogate  min(ρ,c) / mean(min(ρ,c)) · A
  logclip    — K epochs, surrogate  exp(clip(logρ,±c)) · A   (smooth trunc.)
  ppo_clip   — K epochs, PPO clipped surrogate min(ρ·A, clip(ρ,1±ε)·A)

For "standard", K is forced to 1 so ρ≡1 and IS is a no-op.
For all other modes K epochs of gradient steps are performed per rollout.

Usage
-----
  python train/reinforce_is.py
  python train/reinforce_is.py --env LunarLander-v2 --epochs 10 --n-steps 2048
  python train/reinforce_is.py --modes standard,snis,ppo_clip --timesteps 500000
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ENV          = "LunarLander-v3"
TOTAL_TS     = 500_000
N_STEPS      = 2_048       # steps collected per rollout (π_old batch)
K_EPOCHS     = 10          # gradient epochs per rollout (for reuse modes)
LR           = 3e-4
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2         # PPO clip ε
TRUNC_C      = 5.0         # clip threshold for "truncated" mode
LOGCLIP_C    = 1.0         # log-ratio clip threshold (exp(1) ≈ 2.72)
ENT_BONUS    = 0.01        # entropy regularisation
VF_COEF      = 0.5
MAX_GRAD     = 0.5
SEED         = 0
EVAL_FREQ    = 10          # evaluate every N rollouts
N_EVAL_EPS   = 10
N_GIF_EPS    = 2
BATCH_SIZE   = 64          # mini-batch size within each epoch

ALL_MODES = ("standard", "naive", "is", "snis", "truncated",
             "logclip", "ppo_clip")

COLORS = {
    "standard":  "#1f77b4",
    "naive":     "#e377c2",
    "is":        "#2ca02c",
    "snis":      "#ff7f0e",
    "truncated": "#d62728",
    "logclip":   "#17becf",
    "ppo_clip":  "#9467bd",
}

TRAIN_DIR = os.path.dirname(__file__)
RUNS_DIR  = os.path.join(TRAIN_DIR, "runs", "reinforce_is")


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------
class PolicyNet(nn.Module):
    """Shared-trunk actor-critic for discrete action spaces."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.actor_head  = nn.Linear(hidden, act_dim)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

    def get_dist(self, x: torch.Tensor) -> Categorical:
        logits, _ = self(x)
        return Categorical(logits=logits)

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        logits, values = self(x)
        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, entropy, values


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------
def collect_rollout(
    env: gym.Env,
    policy: PolicyNet,
    n_steps: int,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
) -> dict[str, torch.Tensor]:
    """
    Collect n_steps transitions.  Returns tensors including:
      obs, actions, log_prob_old, returns, advantages, values
    """
    obs_buf   = []
    act_buf   = []
    rew_buf   = []
    val_buf   = []
    lp_buf    = []
    done_buf  = []

    obs, _ = env.reset()
    for _ in range(n_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            dist  = policy.get_dist(obs_t)
            value = policy(obs_t)[1]
        action   = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, reward, term, trunc, _ = env.step(action.item())
        done = term or trunc

        obs_buf.append(obs)
        act_buf.append(action.item())
        rew_buf.append(reward)
        val_buf.append(value.item())
        lp_buf.append(log_prob.item())
        done_buf.append(done)

        obs = next_obs if not done else env.reset()[0]

    # Bootstrap last value
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        last_val = policy(obs_t)[1].item()

    # GAE
    obs_arr  = np.array(obs_buf,  dtype=np.float32)
    act_arr  = np.array(act_buf,  dtype=np.int64)
    rew_arr  = np.array(rew_buf,  dtype=np.float32)
    val_arr  = np.array(val_buf,  dtype=np.float32)
    lp_arr   = np.array(lp_buf,   dtype=np.float32)
    done_arr = np.array(done_buf, dtype=np.float32)

    adv_arr = np.zeros_like(rew_arr)
    gae     = 0.0
    for t in reversed(range(n_steps)):
        next_v   = last_val if t == n_steps - 1 else val_arr[t + 1]
        mask     = 1.0 - done_arr[t]
        delta    = rew_arr[t] + gamma * next_v * mask - val_arr[t]
        gae      = delta + gamma * gae_lambda * mask * gae
        adv_arr[t] = gae

    ret_arr = adv_arr + val_arr

    # Normalise advantages
    adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

    def t(a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=device)

    return {
        "obs":       t(obs_arr),
        "actions":   t(act_arr, dtype=torch.long),
        "log_prob_old": t(lp_arr),
        "returns":   t(ret_arr),
        "advantages": t(adv_arr),
        "values":    t(val_arr),
    }


# ---------------------------------------------------------------------------
# Policy-gradient surrogate loss
# ---------------------------------------------------------------------------
def policy_loss(
    mode: str,
    log_prob_new: torch.Tensor,
    log_prob_old: torch.Tensor,
    adv: torch.Tensor,
    clip_eps: float,
    trunc_c: float,
    logclip_c: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Return (loss, diag).  All modes share the same surrogate template

         L = -(w · A).mean()

    where w differs by mode and gradient flows through w (= exp(logπ-logπ_old)),
    EXCEPT for "standard"/"naive" which use the score-function form
    L = -(logπ · A).mean()  because ρ ≡ 1 (or is *meant* to be ignored).
    """
    log_ratio = (log_prob_new - log_prob_old).clamp(-20.0, 20.0)
    ratio     = torch.exp(log_ratio)

    diag = {
        "ratio_mean":  ratio.mean().item(),
        "ratio_max":   ratio.max().item(),
        "ratio_min":   ratio.min().item(),
        "kl_approx":   (-log_ratio).mean().item(),
    }
    # ESS — only meaningful for IS-based modes
    with torch.no_grad():
        diag["ess"] = float(
            ratio.sum().pow(2) / (ratio.pow(2).sum() * len(ratio) + 1e-8))

    if mode == "standard" or mode == "naive":
        # vanilla REINFORCE  (no IS correction)
        loss = -(log_prob_new * adv).mean()

    elif mode == "is":
        # IS-corrected PG:  E_{π_old}[ρ A]  → grad = E[ρ ∇logπ A]
        loss = -(ratio * adv).mean()

    elif mode == "snis":
        # Self-normalised IS.  Detach the denominator so the gradient
        # is unbiased w.r.t. the normaliser.
        denom = ratio.mean().detach().clamp(min=1e-8)
        w     = ratio / denom
        loss  = -(w * adv).mean()

    elif mode == "truncated":
        rho_t = torch.minimum(
            ratio, torch.full_like(ratio, trunc_c))
        denom = rho_t.mean().detach().clamp(min=1e-8)
        w     = rho_t / denom
        loss  = -(w * adv).mean()

    elif mode == "logclip":
        # Symmetric log-ratio clip  ⇒ ρ̂ ∈ [e^{-c}, e^{c}]
        clipped_lr = log_ratio.clamp(-logclip_c, logclip_c)
        w          = torch.exp(clipped_lr)
        loss       = -(w * adv).mean()

    elif mode == "ppo_clip":
        unclipped = ratio * adv
        clipped   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        loss      = -torch.min(unclipped, clipped).mean()
        diag["clip_frac"] = ((ratio < 1 - clip_eps) |
                              (ratio > 1 + clip_eps)).float().mean().item()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return loss, diag


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
def train_on_batch(
    mode: str,
    policy: PolicyNet,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    k_epochs: int,
    batch_size: int,
    ent_bonus: float,
    vf_coef: float,
    max_grad: float,
    clip_eps: float,
    trunc_c:   float = TRUNC_C,
    logclip_c: float = LOGCLIP_C,
) -> dict[str, float]:
    """Run k_epochs of gradient updates on the collected batch."""
    n = batch["obs"].shape[0]
    epochs = 1 if mode == "standard" else k_epochs

    stats = {"pg_loss": [], "vf_loss": [], "ent": [],
             "ratio_mean": [], "ratio_max": [], "ess": [],
             "kl_approx": [], "clip_frac": []}

    for _ in range(epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            mb = idx[start: start + batch_size]
            obs      = batch["obs"][mb]
            actions  = batch["actions"][mb]
            lp_old   = batch["log_prob_old"][mb]
            adv      = batch["advantages"][mb]
            ret      = batch["returns"][mb]

            log_prob_new, entropy, values = policy.evaluate_actions(obs, actions)

            pg_loss, diag = policy_loss(
                mode, log_prob_new, lp_old, adv,
                clip_eps=clip_eps, trunc_c=trunc_c, logclip_c=logclip_c,
            )

            vf_loss  = F.mse_loss(values, ret)
            ent_loss = -entropy.mean()

            loss = pg_loss + vf_coef * vf_loss + ent_bonus * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad)
            optimizer.step()

            stats["pg_loss"].append(pg_loss.item())
            stats["vf_loss"].append(vf_loss.item())
            stats["ent"].append(entropy.mean().item())
            stats["ratio_mean"].append(diag["ratio_mean"])
            stats["ratio_max"].append(diag["ratio_max"])
            stats["ess"].append(diag["ess"])
            stats["kl_approx"].append(diag["kl_approx"])
            if "clip_frac" in diag:
                stats["clip_frac"].append(diag["clip_frac"])

    return {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(policy: PolicyNet, env_name: str, n_eps: int,
             seed: int, device: torch.device) -> tuple[float, float]:
    env   = gym.make(env_name)
    rews  = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed + ep)
        total  = 0.0
        done   = False
        while not done:
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy.get_dist(obs_t).probs.argmax(dim=-1).item()
            obs, r, term, trunc, _ = env.step(action)
            total += r
            done = term or trunc
        rews.append(total)
    env.close()
    return float(np.mean(rews)), float(np.std(rews))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class ReinforceISTrainer:
    def __init__(
        self,
        env_name: str,
        mode: str,
        total_timesteps: int,
        n_steps: int          = N_STEPS,
        k_epochs: int         = K_EPOCHS,
        lr: float             = LR,
        gamma: float          = GAMMA,
        gae_lambda: float     = GAE_LAMBDA,
        clip_eps: float       = CLIP_EPS,
        ent_bonus: float      = ENT_BONUS,
        vf_coef: float        = VF_COEF,
        max_grad: float       = MAX_GRAD,
        batch_size: int       = BATCH_SIZE,
        seed: int             = SEED,
        eval_freq: int        = EVAL_FREQ,
        n_eval_episodes: int  = N_EVAL_EPS,
        group_dir: str | None = None,
    ):
        self.env_name        = env_name
        self.mode            = mode
        self.total_timesteps = total_timesteps
        self.n_steps         = n_steps
        self.k_epochs        = k_epochs
        self.lr              = lr
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_eps        = clip_eps
        self.ent_bonus       = ent_bonus
        self.vf_coef         = vf_coef
        self.max_grad        = max_grad
        self.batch_size      = batch_size
        self.seed            = seed
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.label           = f"REINFORCE-{mode.upper()}"
        self.history: dict[str, list] = {
            "timestep": [], "eval_mean": [], "eval_std": [],
            "pg_loss":  [], "vf_loss":   [], "entropy":  [],
            "ratio_mean": [], "ratio_max": [],
            "ess": [], "kl_approx": [], "clip_frac": [],
        }
        self.device = torch.device("cpu")

        if group_dir is not None:
            self.output_dir = os.path.join(group_dir, mode)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(RUNS_DIR, mode, env_name, ts)
        os.makedirs(self.output_dir, exist_ok=True)

        self._policy: PolicyNet | None = None

    # ------------------------------------------------------------------
    def train(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        env = RecordEpisodeStatistics(gym.make(self.env_name))
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        policy    = PolicyNet(obs_dim, act_dim).to(self.device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.lr, eps=1e-5)
        self._policy = policy

        total_steps = 0
        n_rollouts  = self.total_timesteps // self.n_steps
        pbar        = tqdm(total=self.total_timesteps, desc=self.label,
                           unit="step", dynamic_ncols=True)
        eval_rollout = 0

        for rollout_idx in range(n_rollouts):
            # ── Collect rollout with current π ──────────────────────────
            batch = collect_rollout(
                env, policy, self.n_steps, self.device,
                self.gamma, self.gae_lambda)
            total_steps += self.n_steps
            pbar.update(self.n_steps)

            # ── K-epoch gradient updates ─────────────────────────────────
            stats = train_on_batch(
                mode      = self.mode,
                policy    = policy,
                optimizer = optimizer,
                batch     = batch,
                k_epochs  = self.k_epochs,
                batch_size= self.batch_size,
                ent_bonus = self.ent_bonus,
                vf_coef   = self.vf_coef,
                max_grad  = self.max_grad,
                clip_eps  = self.clip_eps,
            )

            self.history["timestep"].append(total_steps)
            self.history["pg_loss"].append(stats["pg_loss"])
            self.history["vf_loss"].append(stats["vf_loss"])
            self.history["entropy"].append(stats["ent"])
            self.history["ratio_mean"].append(stats["ratio_mean"])
            self.history["ratio_max"].append(stats["ratio_max"])
            self.history["ess"].append(stats["ess"])
            self.history["kl_approx"].append(stats["kl_approx"])
            self.history["clip_frac"].append(stats.get("clip_frac", 0.0))

            # ── Periodic eval ───────────────────────────────────────────
            eval_rollout += 1
            if eval_rollout >= self.eval_freq:
                eval_rollout = 0
                mean_r, std_r = evaluate(
                    policy, self.env_name, self.n_eval_episodes,
                    self.seed, self.device)
                self.history["eval_mean"].append(mean_r)
                self.history["eval_std"].append(std_r)
                pbar.set_postfix({"eval": f"{mean_r:.1f}"})

        pbar.close()
        env.close()

        # Final eval
        mean_r, std_r = evaluate(
            policy, self.env_name, self.n_eval_episodes,
            self.seed, self.device)
        print(f"\n[{self.label}] final eval: {mean_r:.1f} ± {std_r:.1f}")

        # Save
        np.savez(os.path.join(self.output_dir, "metrics.npz"),
                 **{k: np.array(v) for k, v in self.history.items()})
        torch.save(policy.state_dict(),
                   os.path.join(self.output_dir, "policy.pt"))
        print(f"  → {self.output_dir}")

    # ------------------------------------------------------------------
    def _load_history(self) -> dict[str, np.ndarray]:
        if self.history.get("timestep"):
            return {k: np.array(v) for k, v in self.history.items()}
        p = os.path.join(self.output_dir, "metrics.npz")
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            return {k: d[k] for k in d.files}
        return {}

    # ------------------------------------------------------------------
    def plot_results(self) -> None:
        h     = self._load_history()
        color = COLORS.get(self.mode, "steelblue")
        ts    = np.array(h.get("timestep", []), dtype=np.int64)
        eval_ts = np.linspace(0, ts[-1] if len(ts) else 0,
                              len(h.get("eval_mean", [])))

        panels = [
            ("eval",        "Eval Reward",      eval_ts,
             np.array(h.get("eval_mean", []), dtype=float),
             np.array(h.get("eval_std",  []), dtype=float)),
            ("pg_loss",     "PG Loss",          ts,
             np.array(h.get("pg_loss",   []), dtype=float), None),
            ("vf_loss",     "Value Loss",       ts,
             np.array(h.get("vf_loss",   []), dtype=float), None),
            ("entropy",     "Entropy",          ts,
             np.array(h.get("entropy",   []), dtype=float), None),
            ("ratio_mean",  "Mean ρ",           ts,
             np.array(h.get("ratio_mean",[]), dtype=float), None),
            ("ess",         "ESS (Kish)",       ts,
             np.array(h.get("ess",       []), dtype=float), None),
        ]

        def rwin_std(v, w=5):
            out = np.empty_like(v)
            for i in range(len(v)):
                out[i] = v[max(0, i - w): i + w + 1].std()
            return out

        cols = 3
        rows = (len(panels) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
        axf = np.array(axes).flatten()

        for ax, (_, title, x, y, yerr) in zip(axf, panels):
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Timesteps", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            if len(x) == 0 or len(y) == 0:
                continue
            msk = ~np.isnan(y)
            if not msk.any():
                continue
            ln, = ax.plot(x[msk], y[msk], lw=2 if yerr is not None else 1.5,
                          label=self.label, color=color)
            if yerr is not None:
                ax.fill_between(x[msk], y[msk] - yerr[msk],
                                y[msk] + yerr[msk],
                                alpha=0.2, color=ln.get_color())
            else:
                ax.fill_between(x[msk], y[msk] - rwin_std(y[msk]),
                                y[msk] + rwin_std(y[msk]),
                                alpha=0.15, color=ln.get_color())
            ax.legend(fontsize=7)

        for ax in axf[len(panels):]:
            ax.set_visible(False)

        fig.suptitle(f"{self.env_name} -- {self.label}", fontsize=12)
        fig.tight_layout()
        out = os.path.join(self.output_dir, "learning_curve.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Plot  → {out}")
        plt.close(fig)

    # ------------------------------------------------------------------
    def save_renders(self, n_episodes: int = N_GIF_EPS) -> None:
        if self._policy is None:
            obs_dim = gym.make(self.env_name).observation_space.shape[0]
            act_dim = gym.make(self.env_name).action_space.n
            p = PolicyNet(obs_dim, act_dim).to(self.device)
            p.load_state_dict(torch.load(
                os.path.join(self.output_dir, "policy.pt"),
                map_location=self.device))
            self._policy = p

        rdir = os.path.join(self.output_dir, "renders")
        os.makedirs(rdir, exist_ok=True)
        env = gym.make(self.env_name, render_mode="rgb_array")

        for ep in range(n_episodes):
            frames = []
            obs, _ = env.reset(seed=self.seed + ep)
            done   = False
            while not done:
                frames.append(env.render())
                obs_t  = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action = self._policy.get_dist(obs_t).probs.argmax(dim=-1).item()
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
            path = os.path.join(rdir, f"ep{ep+1}.gif")
            imageio.mimsave(path, frames, fps=30)
            print(f"  GIF   → {path}")
        env.close()


# ---------------------------------------------------------------------------
# Overlay plot
# ---------------------------------------------------------------------------
def plot_overlay(trainers: list[ReinforceISTrainer], save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    PANELS = [
        ("eval",       "Eval Reward"),
        ("pg_loss",    "PG Loss"),
        ("vf_loss",    "Value Loss"),
        ("entropy",    "Entropy"),
        ("ratio_mean", "Mean ρ"),
        ("ess",        "ESS (Kish)"),
    ]

    def rwin_std(v, w=5):
        out = np.empty_like(v)
        for i in range(len(v)):
            out[i] = v[max(0, i - w): i + w + 1].std()
        return out

    cols = 3
    rows = (len(PANELS) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows))
    axf      = np.array(axes).flatten()
    env_name = trainers[0].env_name if trainers else ""
    fig.suptitle(f"REINFORCE IS Comparison  ({env_name})", fontsize=13, y=1.01)

    for ax, (key, title) in zip(axf, PANELS):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Timesteps", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        for t in trainers:
            h     = t._load_history()
            color = COLORS.get(t.mode, "steelblue")
            ts    = np.array(h.get("timestep", []), dtype=np.int64)

            if key == "eval":
                y = np.array(h.get("eval_mean", []), dtype=float)
                e = np.array(h.get("eval_std",  []), dtype=float)
                if len(y) == 0:
                    continue
                x_ev = np.linspace(0, ts[-1] if len(ts) else 0, len(y))
                msk  = ~np.isnan(y)
                ln,  = ax.plot(x_ev[msk], y[msk], lw=2,
                               label=t.label, color=color)
                ax.fill_between(x_ev[msk], y[msk] - e[msk],
                                y[msk] + e[msk],
                                alpha=0.15, color=ln.get_color())
            else:
                y = np.array(h.get(key, []), dtype=float)
                if len(y) == 0 or len(ts) == 0:
                    continue
                msk = ~np.isnan(y)
                if not msk.any():
                    continue
                ln, = ax.plot(ts[msk], y[msk], lw=1.5,
                              label=t.label, color=color)
                ax.fill_between(ts[msk],
                                y[msk] - rwin_std(y[msk]),
                                y[msk] + rwin_std(y[msk]),
                                alpha=0.15, color=ln.get_color())

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    for ax in axf[len(PANELS):]:
        ax.set_visible(False)

    fig.tight_layout()
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path   = os.path.join(
        save_dir, f"reinforce_is_overlay__{env_name}__{ts_str}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nOverlay → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stats table
# ---------------------------------------------------------------------------
def _print_stats(trainers: list[ReinforceISTrainer]) -> None:
    col_w  = 26
    header = (f"{'Mode':<{col_w}} {'Mean(last 20%)':>14} {'Std':>8}"
              f" {'Peak':>10} {'#Evals':>7}")
    sep    = "=" * len(header)
    print(f"\n{sep}")
    print(f"  REINFORCE IS Comparison -- {trainers[0].env_name}")
    print(sep)
    print(header)
    print("-" * len(header))
    for t in trainers:
        h = t._load_history()
        y = np.array(h.get("eval_mean", []), dtype=float)
        if len(y) == 0:
            print(f"{t.label:<{col_w}} {'N/A':>14}")
            continue
        last_n = max(1, len(y) // 5)
        print(f"{t.label:<{col_w}}"
              f" {y[-last_n:].mean():>14.1f}"
              f" {y[-last_n:].std():>8.1f}"
              f" {y.max():>10.1f}"
              f" {len(y):>7}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------
def run_comparison(
    env_name: str        = ENV,
    total_timesteps: int = TOTAL_TS,
    n_steps: int         = N_STEPS,
    k_epochs: int        = K_EPOCHS,
    lr: float            = LR,
    gamma: float         = GAMMA,
    gae_lambda: float    = GAE_LAMBDA,
    clip_eps: float      = CLIP_EPS,
    ent_bonus: float     = ENT_BONUS,
    vf_coef: float       = VF_COEF,
    batch_size: int      = BATCH_SIZE,
    seed: int            = SEED,
    eval_freq: int       = EVAL_FREQ,
    n_eval_episodes: int = N_EVAL_EPS,
    n_gif_episodes: int  = N_GIF_EPS,
    modes: tuple         = ALL_MODES,
) -> list[ReinforceISTrainer]:
    ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_dir = os.path.join(RUNS_DIR, "comparison", env_name, ts_str)
    os.makedirs(group_dir, exist_ok=True)

    print(f"\nGroup dir   : {group_dir}")
    print(f"Env         : {env_name}")
    print(f"Timesteps   : {total_timesteps:,}")
    print(f"N steps/rollout : {n_steps}")
    print(f"K epochs/rollout: {k_epochs}  (standard always uses 1)")
    print(f"Modes       : {', '.join(modes)}\n")

    trainers = []
    for mode in modes:
        t = ReinforceISTrainer(
            env_name=env_name, mode=mode,
            total_timesteps=total_timesteps,
            n_steps=n_steps, k_epochs=k_epochs,
            lr=lr, gamma=gamma, gae_lambda=gae_lambda,
            clip_eps=clip_eps, ent_bonus=ent_bonus,
            vf_coef=vf_coef, batch_size=batch_size,
            seed=seed, eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            group_dir=group_dir,
        )
        t.train()
        t.plot_results()
        t.save_renders(n_episodes=n_gif_episodes)
        trainers.append(t)

    _print_stats(trainers)
    plot_overlay(
        trainers,
        save_dir=os.path.join(RUNS_DIR, "comparison", env_name))
    return trainers


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",         default=ENV)
    parser.add_argument("--timesteps",   default=TOTAL_TS,    type=int)
    parser.add_argument("--n-steps",     default=N_STEPS,     type=int,
                        help="Steps collected per rollout (π_old batch size)")
    parser.add_argument("--epochs",      default=K_EPOCHS,    type=int,
                        help="Gradient epochs per rollout (reuse modes)")
    parser.add_argument("--lr",          default=LR,          type=float)
    parser.add_argument("--gamma",       default=GAMMA,       type=float)
    parser.add_argument("--gae-lambda",  default=GAE_LAMBDA,  type=float)
    parser.add_argument("--clip-eps",    default=CLIP_EPS,    type=float,
                        help="PPO clip ε (also controls how stale data becomes)")
    parser.add_argument("--ent-bonus",   default=ENT_BONUS,   type=float)
    parser.add_argument("--vf-coef",     default=VF_COEF,     type=float)
    parser.add_argument("--batch-size",  default=BATCH_SIZE,  type=int)
    parser.add_argument("--seed",        default=SEED,        type=int)
    parser.add_argument("--eval-freq",   default=EVAL_FREQ,   type=int,
                        help="Evaluate every N rollouts")
    parser.add_argument("--n-eval-eps",  default=N_EVAL_EPS,  type=int)
    parser.add_argument("--n-gif-eps",   default=N_GIF_EPS,   type=int)
    parser.add_argument("--modes",       default=",".join(ALL_MODES),
                        help="Comma-separated subset of modes to run")
    cli = parser.parse_args()

    run_comparison(
        env_name        = cli.env,
        total_timesteps = cli.timesteps,
        n_steps         = cli.n_steps,
        k_epochs        = cli.epochs,
        lr              = cli.lr,
        gamma           = cli.gamma,
        gae_lambda      = cli.gae_lambda,
        clip_eps        = cli.clip_eps,
        ent_bonus       = cli.ent_bonus,
        vf_coef         = cli.vf_coef,
        batch_size      = cli.batch_size,
        seed            = cli.seed,
        eval_freq       = cli.eval_freq,
        n_eval_episodes = cli.n_eval_eps,
        n_gif_episodes  = cli.n_gif_eps,
        modes           = tuple(cli.modes.split(",")),
    )
