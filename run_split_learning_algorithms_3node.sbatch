#!/bin/bash -l
#SBATCH --job-name=sl_algo_3node
#SBATCH --partition=compute
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=sl_algo_3node_%j.out

set -euo pipefail

ROOT="$SLURM_SUBMIT_DIR"

source /data/datasets/$USER/venv_split_algos/bin/activate
cd "$ROOT"

export PYTHONPATH="$ROOT"

# Base cache roots on scratch
TORCH_BASE="/scratch-beegfs/datasets/$USER/.cache/torch"
HF_BASE="/scratch-beegfs/datasets/$USER/.cache/huggingface"
FLWR_BASE="/scratch-beegfs/datasets/$USER/.cache/flwr_datasets"

mkdir -p "$TORCH_BASE" "$HF_BASE" "$FLWR_BASE"

readarray -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
SERVER_NODE="${HOSTS[0]}"
CLIENT0_NODE="${HOSTS[1]}"
CLIENT1_NODE="${HOSTS[2]}"

PORT=8080
SERVER_ADDR="${SERVER_NODE}:${PORT}"

echo "SERVER_NODE=$SERVER_NODE"
echo "CLIENT0_NODE=$CLIENT0_NODE"
echo "CLIENT1_NODE=$CLIENT1_NODE"
echo "SERVER_ADDR=$SERVER_ADDR"

# Server on node 1
srun -N1 -n1 -w "$SERVER_NODE" --exclusive bash -lc "
  source /data/datasets/$USER/venv_split_algos/bin/activate
  cd '$ROOT'
  export PYTHONPATH='$ROOT'

  export TORCH_HOME='$TORCH_BASE/server'
  export HF_HOME='$HF_BASE/server'
  export HF_DATASETS_CACHE='$HF_BASE/server/datasets'
  export COLEXT_DATA_HOME_FOLDER='$FLWR_BASE/server'
  mkdir -p \"\$TORCH_HOME\" \"\$HF_HOME\" \"\$HF_DATASETS_CACHE\" \"\$COLEXT_DATA_HOME_FOLDER\"

  export CUDA_VISIBLE_DEVICES=0
  export HYDRA_FULL_ERROR=1
  export COLEXT_N_CLIENTS=2

  echo SERVER host: \$(hostname)
  nvidia-smi -L

  python -u scripts/py/run_server.py \
    general.num_rounds=2 \
    partitioning.num_partitions=2 \
    strategy_config.fraction_fit=1.0 \
    strategy_config.fraction_evaluate=1.0
" > "$ROOT/server_${SLURM_JOB_ID}.log" 2>&1 &

sleep 20

# Client 0 on node 2
srun -N1 -n1 -w "$CLIENT0_NODE" --exclusive bash -lc "
  source /data/datasets/$USER/venv_split_algos/bin/activate
  cd '$ROOT'
  export PYTHONPATH='$ROOT'

  export TORCH_HOME='$TORCH_BASE/client0'
  export HF_HOME='$HF_BASE/client0'
  export HF_DATASETS_CACHE='$HF_BASE/client0/datasets'
  export COLEXT_DATA_HOME_FOLDER='$FLWR_BASE/client0'
  mkdir -p \"\$TORCH_HOME\" \"\$HF_HOME\" \"\$HF_DATASETS_CACHE\" \"\$COLEXT_DATA_HOME_FOLDER\"

  export CUDA_VISIBLE_DEVICES=0
  export HYDRA_FULL_ERROR=1
  export COLEXT_SERVER_ADDRESS='${SERVER_ADDR}'
  export COLEXT_CLIENT_ID=0
  export COLEXT_DEVICE_TYPE=weak_device

  echo CLIENT0 host: \$(hostname)
  nvidia-smi -L

  python -u scripts/py/run_client.py \
    general.num_rounds=2 \
    partitioning.num_partitions=2 \
    strategy_config.fraction_fit=1.0 \
    strategy_config.fraction_evaluate=1.0
" > "$ROOT/client0_${SLURM_JOB_ID}.log" 2>&1 &

# Client 1 on node 3
srun -N1 -n1 -w "$CLIENT1_NODE" --exclusive bash -lc "
  source /data/datasets/$USER/venv_split_algos/bin/activate
  cd '$ROOT'
  export PYTHONPATH='$ROOT'

  export TORCH_HOME='$TORCH_BASE/client1'
  export HF_HOME='$HF_BASE/client1'
  export HF_DATASETS_CACHE='$HF_BASE/client1/datasets'
  export COLEXT_DATA_HOME_FOLDER='$FLWR_BASE/client1'
  mkdir -p \"\$TORCH_HOME\" \"\$HF_HOME\" \"\$HF_DATASETS_CACHE\" \"\$COLEXT_DATA_HOME_FOLDER\"

  export CUDA_VISIBLE_DEVICES=0
  export HYDRA_FULL_ERROR=1
  export COLEXT_SERVER_ADDRESS='${SERVER_ADDR}'
  export COLEXT_CLIENT_ID=1
  export COLEXT_DEVICE_TYPE=strong_device

  echo CLIENT1 host: \$(hostname)
  nvidia-smi -L

  python -u scripts/py/run_client.py \
    general.num_rounds=2 \
    partitioning.num_partitions=2 \
    strategy_config.fraction_fit=1.0 \
    strategy_config.fraction_evaluate=1.0
" > "$ROOT/client1_${SLURM_JOB_ID}.log" 2>&1 &

wait
echo DONE
