"""
IS gradient estimator comparison trainer.

Two groups, run independently — same 5 IS weight variants each, same n_epochs:

  PPO        (--algo ppo):       clipped objective  min(w·A, clip(w, 1±ε)·A)
  PPO-NoClip (--algo reinforce): unclipped objective  w·A  (no PPO clip)

Usage
-----
  # PPO (with clip) — stressed IS: clip=0.3, 15 epochs
  python train/train.py --algo ppo --compare --env HalfCheetah-v4 \
      --timesteps 1000000 --n-envs 4 --clip-range 0.3 --n-epochs 15

  # PPO-NoClip (without clip) — same stress settings
  python train/train.py --algo reinforce --compare --env HalfCheetah-v4 \
      --timesteps 1000000 --n-envs 4 --clip-range 0.3 --n-epochs 15

  # Single run
  python train/train.py --algo ppo --mode snis --env HalfCheetah-v4
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from estimators import (
    SNISPPO, DefensivePPO, PowerPPO, TruncatedPPO,
    VanillaPG, VanillaPGSNIS, VanillaPGDefensive, VanillaPGPower, VanillaPGTruncated,
    PPO_MODES, REINFORCE_MODES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ENV_NAMES  = ["Pendulum-v1", "HalfCheetah-v4", "Hopper-v4",
               "MountainCarContinuous-v0", "LunarLander-v3"]
ALGO_NAMES = ["ppo", "reinforce"]

TRAIN_DIR = os.path.dirname(__file__)
RUNS_DIR  = os.path.join(TRAIN_DIR, "runs")

COLORS = {
    "standard":   "#1f77b4",  # blue
    "snis":       "#ff7f0e",  # orange
    "defensive":  "#2ca02c",  # green
    "power":      "#9467bd",  # purple
    "truncated":  "#d62728",  # red
}


ALL_METRIC_KEYS = [
    "rollout/ep_rew_mean", "rollout/ep_len_mean",
    "train/policy_gradient_loss", "train/policy_loss",
    "train/value_loss", "train/entropy_loss",
    "train/approx_kl", "train/clip_fraction",
    "train/ratio_mean", "train/ratio_std",
    "train/std", "train/learning_rate", "time/fps",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _label(mode: str, algo: str) -> str:
    prefix = "PPO" if algo == "ppo" else "NoClip"
    return {
        "standard":  f"{prefix}-Standard",
        "snis":      f"{prefix}-SNIS",
        "defensive": f"{prefix}-Defensive",
        "power":     f"{prefix}-Power(β=0.5)",
        "truncated": f"{prefix}-Truncated(c=2)",
    }[mode]


def _color(mode: str) -> str:
    return COLORS.get(mode, "steelblue")


def _modes_for(algo: str) -> list[str]:
    return PPO_MODES if algo == "ppo" else REINFORCE_MODES


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def make_model(algo: str, mode: str, env, seed: int,
               clip_range: float = 0.2, n_epochs: int = 10):
    kw = dict(verbose=0, seed=seed, clip_range=clip_range, n_epochs=n_epochs)
    if algo == "ppo":
        if mode == "standard": return PPO("MlpPolicy", env, **kw)
        _cls = {
            "snis":      SNISPPO,
            "defensive": DefensivePPO,
            "power":     PowerPPO,
            "truncated": TruncatedPPO,
        }
        return _cls[mode]("MlpPolicy", env, **kw)
    elif algo == "reinforce":
        if mode == "standard":  return VanillaPG("MlpPolicy", env, **kw)
        if mode == "snis":      return VanillaPGSNIS("MlpPolicy", env, **kw)
        if mode == "defensive": return VanillaPGDefensive("MlpPolicy", env, **kw)
        if mode == "power":     return VanillaPGPower("MlpPolicy", env, **kw)
        if mode == "truncated": return VanillaPGTruncated("MlpPolicy", env, **kw)
        raise ValueError(f"Unknown VanillaPG mode: {mode}")
    raise ValueError(f"Unknown algo: {algo}")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.history: dict[str, list] = {"timestep": []}

    def _on_step(self) -> bool:
        logged = self.model.logger.name_to_value
        if not logged:
            return True
        self.history["timestep"].append(self.num_timesteps)
        for key in ALL_METRIC_KEYS:
            self.history.setdefault(key, [])
            self.history[key].append(logged.get(key, float("nan")))
        return True


class GradientStatsCallback(BaseCallback):
    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = log_freq
        self._hooks: list = []
        self._buf: list[torch.Tensor] = []
        self._count = 0
        self.history: dict[str, list] = {
            "timestep": [], "grad/actor_mean": [], "grad/actor_var": []
        }

    def _on_training_start(self) -> None:
        actor = getattr(self.model.policy, "mlp_extractor", None)
        if actor is None:
            return
        def _hook(g: torch.Tensor) -> None:
            if g is not None:
                self._buf.append(g.detach().cpu().flatten())
        for p in actor.parameters():
            if p.requires_grad:
                self._hooks.append(p.register_hook(_hook))

    def _on_step(self) -> bool:
        if not self._buf:
            return True
        self._count += 1
        if self._count % self.log_freq == 0:
            g = torch.cat(self._buf)
            self.history["timestep"].append(self.num_timesteps)
            self.history["grad/actor_mean"].append(g.abs().mean().item())
            self.history["grad/actor_var"].append(g.var().item())
            self._buf.clear()
        return True

    def _on_training_end(self) -> None:
        for h in self._hooks:
            h.remove()


class TqdmCallback(BaseCallback):
    def __init__(self, total: int, label: str):
        super().__init__()
        self._total = total
        self._label = label
        self.pbar   = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self._total, desc=self._label, unit="step")

    def _on_step(self) -> bool:
        self.pbar.update(self.training_env.num_envs)
        if self.model.ep_info_buffer:
            r = np.mean([e["r"] for e in self.model.ep_info_buffer])
            self.pbar.set_postfix({"rew": f"{r:.1f}"})
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(
        self,
        env_name: str,
        algo_name: str = "ppo",
        mode: str = "standard",
        total_timesteps: int = 500_000,
        n_envs: int = 4,
        eval_freq: int = 1_000,
        n_eval_episodes: int = 10,
        seed: int = 0,
        group_dir: str | None = None,
        clip_range: float = 0.2,
        n_epochs: int = 10,
    ):
        self.env_name        = env_name
        self.algo_name       = algo_name.lower()
        self.mode            = mode
        self.total_timesteps = total_timesteps
        self.n_envs          = n_envs
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed            = seed
        self.clip_range      = clip_range
        self.n_epochs        = n_epochs
        self.label           = _label(mode, algo_name)

        if group_dir is not None:
            self.output_dir = os.path.join(group_dir, mode)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(RUNS_DIR, algo_name, mode, env_name, ts)

        os.makedirs(self.output_dir, exist_ok=True)
        self._model = None
        self.metrics: dict = {}

    # ------------------------------------------------------------------
    def train(self) -> None:
        train_env = make_vec_env(self.env_name, n_envs=self.n_envs, seed=self.seed)
        eval_env  = make_vec_env(self.env_name, n_envs=1, seed=self.seed + 1)

        model = make_model(self.algo_name, self.mode, train_env, self.seed,
                           clip_range=self.clip_range, n_epochs=self.n_epochs)

        eval_cb    = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.output_dir, "best_model"),
            log_path=self.output_dir,
            eval_freq=self.eval_freq,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            verbose=0,
        )
        metrics_cb = MetricsCallback()
        grad_cb    = GradientStatsCallback(log_freq=100)
        tqdm_cb    = TqdmCallback(self.total_timesteps, self.label)

        model.learn(
            total_timesteps=self.total_timesteps,
            callback=[eval_cb, metrics_cb, grad_cb, tqdm_cb],
        )

        merged = dict(metrics_cb.history)
        merged["grad/timestep"]   = grad_cb.history["timestep"]
        merged["grad/actor_mean"] = grad_cb.history["grad/actor_mean"]
        merged["grad/actor_var"]  = grad_cb.history["grad/actor_var"]

        self.metrics = merged
        self._model  = model
        np.savez(os.path.join(self.output_dir, "metrics.npz"),
                 **{k: np.array(v) for k, v in merged.items()})
        model.save(os.path.join(self.output_dir, "final_model"))

        mean_r, std_r = evaluate_policy(model, eval_env,
                                        n_eval_episodes=self.n_eval_episodes)
        print(f"\n[{self.label}] final eval: {mean_r:.1f} +/- {std_r:.1f}")
        print(f"  -> {self.output_dir}")

    # ------------------------------------------------------------------
    def _load_eval(self):
        p = os.path.join(self.output_dir, "evaluations.npz")
        if not os.path.exists(p):
            return None
        d = np.load(p)
        return d["timesteps"], d["results"].mean(1), d["results"].std(1)

    def _load_metrics(self):
        p = os.path.join(self.output_dir, "metrics.npz")
        return dict(np.load(p)) if os.path.exists(p) else {}

    # ------------------------------------------------------------------
    def plot_results(self) -> None:
        m     = self._load_metrics()
        ts    = m.get("timestep", np.array([]))
        ev    = self._load_eval()
        gts   = m.get("grad/timestep", np.array([]))
        color = _color(self.mode)
        pg_key = ("train/policy_gradient_loss" if self.algo_name == "ppo"
                  else "train/policy_loss")

        panels = [
            ("eval",               "Eval Reward"),
            (pg_key,               "PG Loss"),
            ("train/value_loss",   "Value Loss"),
            ("train/entropy_loss", "Entropy Loss"),
            ("grad/actor_var",     "Gradient Variance"),
            ("train/ratio_mean",   "Mean IS Weight"),
        ]

        def masked(key, t=None):
            if key not in m: return None, None
            v = np.array(m[key], dtype=float)
            msk = ~np.isnan(v)
            t = t if t is not None else ts
            if len(t) != len(v): return None, None
            return t[msk], v[msk]

        def rwin_std(v, w=5):
            out = np.empty_like(v)
            for i in range(len(v)):
                out[i] = v[max(0, i-w): i+w+1].std()
            return out

        cols = 3
        rows = (len(panels) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 4*rows))
        axf = np.array(axes).flatten()

        for ax, (key, title) in zip(axf, panels):
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Timesteps", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            if key == "eval":
                if ev is not None:
                    et, em, es = ev
                    ln, = ax.plot(et, em, lw=2, label=self.label, color=color)
                    ax.fill_between(et, em-es, em+es, alpha=0.2, color=ln.get_color())
                    ax.legend(fontsize=7)
            elif key == "grad/actor_var":
                x, y = masked(key, gts)
                if x is not None:
                    ln, = ax.plot(x, y, lw=1.5, label=self.label, color=color)
                    ax.fill_between(x, y-rwin_std(y), y+rwin_std(y),
                                    alpha=0.15, color=ln.get_color())
                    ax.legend(fontsize=7)
            else:
                x, y = masked(key)
                if x is not None:
                    ax.plot(x, y, lw=1.5, label=self.label, color=color)
                    ax.legend(fontsize=7)

        for ax in axf[len(panels):]:
            ax.set_visible(False)

        fig.suptitle(f"{self.env_name} -- {self.label}", fontsize=12)
        fig.tight_layout()
        out = os.path.join(self.output_dir, "learning_curve.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Plot -> {out}")
        plt.close(fig)

    # ------------------------------------------------------------------
    def save_renders(self, n_episodes: int = 3) -> None:
        import gymnasium as gym
        _cls_map = {
            "standard":  PPO,
            "snis":      SNISPPO,
            "defensive": DefensivePPO,
            "power":     PowerPPO,
            "truncated": TruncatedPPO,
        }
        cls   = _cls_map.get(self.mode, PPO)
        model = self._model or cls.load(os.path.join(self.output_dir, "final_model"))
        rdir  = os.path.join(self.output_dir, "renders")
        os.makedirs(rdir, exist_ok=True)
        env = gym.make(self.env_name, render_mode="rgb_array")
        for ep in range(n_episodes):
            frames, obs, done = [], env.reset(seed=self.seed + ep)[0], False
            while not done:
                frames.append(env.render())
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                done = term or trunc
            path = os.path.join(rdir, f"ep{ep+1}.gif")
            imageio.mimsave(path, frames, fps=30)
            print(f"  GIF  -> {path}")
        env.close()


# ---------------------------------------------------------------------------
# Comparison runner (sequential)
# ---------------------------------------------------------------------------
def run_comparison(
    env_name: str,
    algo_name: str,
    total_timesteps: int,
    n_envs: int,
    seed: int,
    clip_range: float = 0.2,
    n_epochs: int = 10,
) -> list[Trainer]:
    modes     = _modes_for(algo_name)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_dir = os.path.join(RUNS_DIR, "comparison", algo_name, env_name, ts)
    os.makedirs(group_dir, exist_ok=True)
    print(f"Group dir: {group_dir}")
    print(f"Algo: {algo_name.upper()}  |  Modes: {modes}"
          f"  |  clip_range={clip_range}  n_epochs={n_epochs}\n")

    trainers = []
    for m in modes:
        t = Trainer(
            env_name=env_name, algo_name=algo_name, mode=m,
            total_timesteps=total_timesteps, n_envs=n_envs,
            seed=seed, group_dir=group_dir,
            clip_range=clip_range, n_epochs=n_epochs,
        )
        t.train()
        t.plot_results()
        t.save_renders(n_episodes=3)
        trainers.append(t)

    _print_stats_table(trainers)
    return trainers


def _print_stats_table(trainers: list) -> None:
    col_w  = 24
    header = (f"{'Mode':<{col_w}} {'Mean Reward':>12} {'Std':>8}"
              f" {'Max Reward':>12} {'#Evals':>7}")
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(f"  {trainers[0].algo_name.upper()} -- {trainers[0].env_name}")
    print(sep)
    print(header)
    print("-" * len(header))
    for t in trainers:
        ev = t._load_eval()
        if ev is None:
            print(f"{t.label:<{col_w}} {'N/A':>12}")
            continue
        _, em, _ = ev
        last_n = max(1, len(em) // 5)
        print(f"{t.label:<{col_w}} {em[-last_n:].mean():>12.1f}"
              f" {em[-last_n:].std():>8.1f} {em.max():>12.1f} {len(em):>7}")
    print(sep + "\n")


def plot_overlay(trainers: list, save_dir: str) -> None:
    """Produces one overlay figure comparing all IS estimators."""
    os.makedirs(save_dir, exist_ok=True)
    algo   = trainers[0].algo_name
    pg_key = ("train/policy_gradient_loss" if algo == "ppo" else "train/policy_loss")

    PANELS = [
        ("eval",               "Eval Reward"),
        (pg_key,               "PG Loss"),
        ("train/value_loss",   "Value Loss"),
        ("train/entropy_loss", "Entropy Loss"),
        ("grad/actor_var",     "Gradient Variance"),
        ("train/ratio_mean",   "Mean IS Weight"),
    ]

    def rwin_std(v, w=5):
        out = np.empty_like(v)
        for i in range(len(v)):
            out[i] = v[max(0, i-w): i+w+1].std()
        return out

    def _draw_figure(subset: list, title_suffix: str, fname: str) -> None:
        cols = 3
        rows = (len(PANELS) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 5*rows))
        axf = np.array(axes).flatten()
        fig.suptitle(
            f"{trainers[0].env_name} -- {algo.upper()} [{title_suffix}]",
            fontsize=12, y=1.01,
        )

        for ax, (key, title) in zip(axf, PANELS):
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Timesteps", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            for t in subset:
                m     = t._load_metrics()
                color = _color(t.mode)
                ls    = "-"
                label = t.label

                if key == "eval":
                    ev = t._load_eval()
                    if ev is None: continue
                    et, em, es = ev
                    ln, = ax.plot(et, em, lw=2, label=label, color=color, ls=ls)
                    ax.fill_between(et, em-es, em+es, alpha=0.15, color=ln.get_color())
                elif key == "grad/actor_var":
                    gts = m.get("grad/timestep", np.array([]))
                    if key not in m or len(gts) == 0: continue
                    y = np.array(m[key], dtype=float)
                    msk = ~np.isnan(y)
                    if not msk.any(): continue
                    ln, = ax.plot(gts[msk], y[msk], lw=1.5, label=label, color=color, ls=ls)
                    ax.fill_between(gts[msk], y[msk]-rwin_std(y[msk]),
                                    y[msk]+rwin_std(y[msk]), alpha=0.15, color=ln.get_color())
                else:
                    tss = m.get("timestep", np.array([]))
                    if key not in m or len(tss) == 0: continue
                    y = np.array(m[key], dtype=float)
                    msk = ~np.isnan(y)
                    if not msk.any(): continue
                    ln, = ax.plot(tss[msk], y[msk], lw=1.5, label=label, color=color, ls=ls)
                    ax.fill_between(tss[msk], y[msk]-rwin_std(y[msk]),
                                    y[msk]+rwin_std(y[msk]), alpha=0.15, color=ln.get_color())

            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7)

        for ax in axf[len(PANELS):]:
            ax.set_visible(False)

        fig.tight_layout()
        path = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\nOverlay -> {path}")
        plt.show()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = trainers[0].env_name
    suffix = "PPO" if algo == "ppo" else "NoClip"
    _draw_figure(trainers, suffix, f"overlay__{env}__{suffix.lower()}__{ts}.png")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IS gradient estimator comparison -- PPO (clipped) vs PPO-NoClip (unclipped)")
    parser.add_argument("--env",        default="HalfCheetah-v4", choices=ENV_NAMES)
    parser.add_argument("--algo",       default="ppo",            choices=ALGO_NAMES,
                        help="ppo = clipped objective; reinforce = same but no PPO clip")
    parser.add_argument("--mode",       default=None,
                        help="Single-run mode: standard/snis/defensive/power/truncated")
    parser.add_argument("--timesteps",  default=1_000_000, type=int)
    parser.add_argument("--n-envs",     default=4,         type=int)
    parser.add_argument("--seed",       default=0,         type=int)
    parser.add_argument("--clip-range", default=0.3,       type=float,
                        help="PPO clip epsilon (default 0.3 to stress IS ratio)")
    parser.add_argument("--n-epochs",   default=15,        type=int,
                        help="Update epochs per rollout. Use 1 for VanillaPG.")
    parser.add_argument("--compare",    action="store_true",
                        help="Run all modes sequentially and produce overlay plot")
    args = parser.parse_args()

    if args.compare:
        trainers = run_comparison(
            env_name=args.env,
            algo_name=args.algo,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            clip_range=args.clip_range,
            n_epochs=args.n_epochs,
        )
        plot_overlay(
            trainers,
            save_dir=os.path.join(RUNS_DIR, "comparison", args.algo, args.env),
        )
    else:
        mode = args.mode or "standard"
        t = Trainer(
            env_name=args.env,
            algo_name=args.algo,
            mode=mode,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            clip_range=args.clip_range,
            n_epochs=args.n_epochs,
        )
        t.train()
        t.plot_results()
        t.save_renders(n_episodes=3)
