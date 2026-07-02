#!/bin/bash
#SBATCH --job-name=rvml-encoder
#SBATCH --partition=gpu            # <-- ADJUST: RHUL GPU partition name (e.g. gpu, gpu-a100)
#SBATCH --gres=gpu:1               # 1 GPU is plenty — the resnet encoder is small
#SBATCH --cpus-per-task=8          # dataloader workers + host-side work
#SBATCH --mem=32G                  # 500k cache (~3 GB) is loaded whole into RAM
#SBATCH --time=08:00:00            # generous; a GPU run should finish well inside this
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
##SBATCH --account=YOUR_ACCOUNT    # <-- UNCOMMENT + set if the cluster requires an allocation
##SBATCH --mail-type=END,FAIL      # <-- optional
##SBATCH --mail-user=you@rhul.ac.uk

# -------------------------------------------------------------------------
# RV-ML encoder training on the RHUL GPU partition.
# Two-phase run: pretrain 300 ep on the 500k synthetic cache -> finetune 100
# ep on real data. Writes checkpoints/resnet_{pretrain,finetune}_{best,last}.pt.
#
# Submit from the repo root:   sbatch slurm/train_encoder.sbatch
# -------------------------------------------------------------------------

set -euo pipefail

# Run from the directory the job was submitted from (repo root).
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm/logs checkpoints

# ---- Environment -------------------------------------------------------
# Option A (modules): load a CUDA-enabled toolchain. ADJUST module names to
# whatever `module avail` shows on the RHUL cluster.
# module purge
# module load Python/3.13   CUDA/12.x

# Option B (venv): reuse the repo's virtualenv. Create once on the cluster with
#   python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# making sure the installed torch is a CUDA build (not the CPU wheel).
source .venv/bin/activate

export MPLBACKEND=Agg              # headless plotting on the compute node
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ---- Sanity: confirm the GPU is actually visible to torch ---------------
nvidia-smi || true
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
assert torch.cuda.is_available(), "No CUDA device visible — check --gres / torch build"
PY

# ---- Preconditions: cache + real data must be present on the cluster ----
test -f data/pretrain_cache_v3.pt || { echo "MISSING data/pretrain_cache_v3.pt (rsync it over)"; exit 1; }
test -f data/dataset_stats.json   || { echo "MISSING data/ (real .tbl + labels/splits/stats needed for finetune)"; exit 1; }

# ---- Train --------------------------------------------------------------
srun python train.py \
    --pretrain-epochs 300 \
    --finetune-epochs 100 \
    --pretrain-cache data/pretrain_cache_v3.pt \
    --arch resnet \
    --device cuda \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --batch-size 128

echo "Done. Best real checkpoint: checkpoints/resnet_finetune_best.pt"
