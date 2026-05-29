#!/usr/bin/env zsh
# ============================================================
# REINFORCE IS comparison sweep - 4 requested envs x 7 modes x 1M steps
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
echo " REINFORCE IS Requested-Env Sweep"
echo " Modes    : $MODES"
echo " Steps    : $TIMESTEPS"
echo " K epochs : $EPOCHS"
echo " N steps  : $N_STEPS"
echo " Started  : $(date)"
echo "=========================================="

# ------------------------------------------------------------
# 1) HalfCheetah-v5
# ------------------------------------------------------------
echo "\n[1/4] HalfCheetah-v5"
python train/reinforce_is.py \
  --env HalfCheetah-v5 \
  --timesteps $TIMESTEPS \
  --n-steps $N_STEPS \
  --epochs $EPOCHS \
  --modes $MODES

# ------------------------------------------------------------
# 2) FrozenLake-v1
# ------------------------------------------------------------
echo "\n[2/4] FrozenLake-v1"
python train/reinforce_is.py \
  --env FrozenLake-v1 \
  --timesteps $TIMESTEPS \
  --n-steps $N_STEPS \
  --epochs $EPOCHS \
  --modes $MODES

# ------------------------------------------------------------
# 3) FetchPush-v3
# ------------------------------------------------------------
echo "\n[3/4] FetchPush-v3"
python train/reinforce_is.py \
  --env FetchPush-v3 \
  --timesteps $TIMESTEPS \
  --n-steps $N_STEPS \
  --epochs $EPOCHS \
  --modes $MODES

# ------------------------------------------------------------
# 4) MontezumaRevenge-v5
# ------------------------------------------------------------
echo "\n[4/4] MontezumaRevenge-v5"
python train/reinforce_is.py \
  --env MontezumaRevenge-v5 \
  --timesteps $TIMESTEPS \
  --n-steps $N_STEPS \
  --epochs $EPOCHS \
  --modes $MODES

echo "\n=========================================="
echo " All done: $(date)"
echo "=========================================="
