#!/usr/bin/env zsh
# ============================================================
# REINFORCE IS comparison sweep — 3 envs × 7 modes × 1M steps
# Run with lid shut via caffeinate -dis
# ============================================================
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

MODES="snis,standard,naive,is,truncated,logclip,ppo_clip"
TIMESTEPS=1000000
EPOCHS=10
N_STEPS=2048

echo "=========================================="
echo " REINFORCE IS Sweep"
echo " Modes    : $MODES"
echo " Steps    : $TIMESTEPS"
echo " K epochs : $EPOCHS"
echo " Started  : $(date)"
echo "=========================================="

# ------------------------------------------------------------
# 1) LunarLander-v3 — sparse+mixed, moderate difficulty
# ------------------------------------------------------------
echo "\n[1/3] LunarLander-v3"
python train/reinforce_is.py \
  --env LunarLander-v3 \
  --timesteps $TIMESTEPS \
  --n-steps $N_STEPS \
  --epochs $EPOCHS \
  --modes $MODES

# ------------------------------------------------------------
# 2) Acrobot-v1 — dense (-1/step), long horizon, hard
#    n-steps=4096 to capture longer episodes
# ------------------------------------------------------------
echo "\n[2/3] Acrobot-v1"
python train/reinforce_is.py \
  --env Acrobot-v1 \
  --timesteps $TIMESTEPS \
  --n-steps 4096 \
  --epochs $EPOCHS \
  --modes $MODES

# ------------------------------------------------------------
# 3) CartPole-v1 — dense (+1/step), easy, NEGATIVE CONTROL
#    expect: IS ≈ standard (no benefit when already fast)
# ------------------------------------------------------------
echo "\n[3/3] CartPole-v1"
python train/reinforce_is.py \
  --env CartPole-v1 \
  --timesteps 300000 \
  --n-steps 2048 \
  --epochs $EPOCHS \
  --modes $MODES

echo "\n=========================================="
echo " All done: $(date)"
echo "=========================================="
