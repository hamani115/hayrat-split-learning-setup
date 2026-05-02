# Hayrat Split Learning Setup

This repository contains a **working Hayrat setup** for running a distributed test of split learning on the Benefit AI Lab cluster using:

- **1 server**
- **2 clients**
- **3 separate `compute` nodes**
- **1 T4 GPU per node**

It includes the code/config needed to run this setup (`conf/`, `scripts/`, `src/`), along with the ready-to-use Slurm script `run_split_learning_algorithms_3node.sbatch`.

Hayrat uses **Slurm**, and jobs should be submitted to compute nodes rather than run on the login node.

## Background and upstream references

This setup is based on the split-learning code path packaged in this repository for Hayrat use. If you want to inspect the original upstream pieces:

- **SplitBud framework:** `sands-lab/splitbud`  
  https://github.com/sands-lab/splitbud?tab=readme-ov-file

- **SplitBud runnable examples:** `BorisRado/split_learning_algorithms`  
  https://github.com/BorisRado/split_learning_algorithms/tree/masteri

This repository is meant to make it easy for the team to **clone, run, and inspect** a Hayrat-tested setup directly.

---

## 1. Clone this repository on Hayrat

Log in to Hayrat, then clone this repository:

```bash
cd /scratch-beegfs/datasets/$USER
git clone https://github.com/hamani115/hayrat-split-learning-setup.git
cd hayrat-split-learning-setup
```

---

## 2. Create and activate the Python environment

Use a dedicated virtual environment:

```bash
conda activate py39
python -m venv /data/datasets/$USER/venv_split_algos
source /data/datasets/$USER/venv_split_algos/bin/activate
```

Install the dependencies:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
```

---

## 3. Sanity-check the environment

Run these checks before submitting jobs:

```bash
python -c "from slbd.server.app import start_server; from slbd.client.app import start_client; print('slbd ok')"
python -c "import src; print('repo src ok')"
```

If both commands print successfully, the environment is ready.

---

## 4. Use the included Slurm script

This repository already includes the Slurm submission script:

```bash
run_split_learning_algorithms_3node.sbatch
```

It is configured for:

* `--partition=compute`
* `--nodes=3`
* `--ntasks=3`
* `--ntasks-per-node=1`

which matches the intended 3-node distributed setup on Hayrat.

---

## 5. Clean old caches if needed

If a previous run failed during dataset download, clean the old caches first:

```bash
rm -rf /scratch-beegfs/datasets/$USER/.cache/huggingface
rm -rf /scratch-beegfs/datasets/$USER/.cache/flwr_datasets
```

This helps avoid broken Hugging Face `.incomplete` cache files from affecting future runs.

---

## 6. Submit the job

Submit the distributed run with:

```bash
sbatch run_split_learning_algorithms_3node.sbatch
```

Check the queue with:

```bash
squeue --me
```

Hayrat’s Slurm docs describe `sbatch` submission and `squeue` monitoring for jobs running on compute nodes. ([Hayrat Slurm](https://uob-ai.github.io/slurm.html))

---

## 7. Inspect the main Slurm output

The batch script writes the main output file as:

```bash
sl_algo_3node_<JOBID>.out
```

Example:

```bash
cat sl_algo_3node_23103.out
```

This file shows:

* allocated nodes
* server/client node mapping
* the server address

---

## 8. Inspect the server and client logs

The Slurm script writes separate logs for each process:

* `server_<JOBID>.log`
* `client0_<JOBID>.log`
* `client1_<JOBID>.log`

Inspect them with:

```bash
tail -n 80 server_<JOBID>.log
tail -n 80 client0_<JOBID>.log
tail -n 80 client1_<JOBID>.log
```

---

## 9. What a successful run looks like

### In the server log

You should see:

* Flower server starts
* both clients report properties
* training rounds run successfully
* evaluation rounds run successfully
* no failures
* final summary with loss/accuracy history

Typical successful lines:

```text
fit_round 1: strategy sampled 2 clients (out of 2)
fit_round 1 received 2 results and 0 failures
evaluate_round 1 received 2 results and 0 failures
...
Run finished 2 round(s)
```

### In the client logs

You should see:

* `ChannelConnectivity.READY`
* `Received: get_properties`
* `Received: train message`
* `Received: evaluate message`
* `Disconnect and shut down`

This means the client connected, trained, evaluated, and exited normally.

---

## 10. Tested setup

This setup was successfully tested with:

* **3 separate compute nodes**
* **1 T4 GPU per node**
* **1 server**
* **2 clients**
* **2 rounds**
* **2 IID partitions**
* **heterogeneous clients**

  * `weak_device`
  * `strong_device`

---

## 11. Useful Hayrat commands

Show available partitions:

```bash
sinfo
```

Show detailed node information:

```bash
sinfo -N -l
```

Cancel a stuck job:

```bash
scancel <JOBID>
```

Hayrat documents the `standard`, `compute`, and `gpu` partitions and recommends `sinfo` / `sinfo -N -l` for cluster inspection. ([GitHub][2])

---

## 12. Quick summary

The working flow is:

1. Clone this repo
2. Create and activate `venv_split_algos`
3. Install `requirements.txt`
4. Clean caches if needed
5. Submit `run_split_learning_algorithms_3node.sbatch`
6. Inspect:

   * `sl_algo_3node_<JOBID>.out`
   * `server_<JOBID>.log`
   * `client0_<JOBID>.log`
   * `client1_<JOBID>.log`


