# Dense-Urban GNN–Dueling-DDQN Handover Prototype

This repository is the **code-first Stage A** for a publishable handover study. It models a sparse UE–cell bipartite graph, uses edge-aware graph attention, and trains a parameter-shared Dueling Double-DQN with prioritized replay.

## First correction: define “100 towers” precisely

The supplied target configuration uses **100 radio cells/base stations and 100 UEs** in a 2 km × 2 km dense-urban area. If “100 towers” means 100 physical three-sector sites, the graph has roughly **300 cell nodes**, not 100; do not mix these meanings in a manuscript.

A 100-cell/100-UE network averages only one UE per cell, so load balancing may be weak. Keep it as the requested minimum, then test **100 cells × 300 UEs** and **100 cells × 500 UEs** as load-stress cases.

## Why this action design scales

A single global DQN over all assignments is impossible: even 8 choices for each of 100 UEs gives `8^100` joint actions. This project instead:

1. keeps the serving cell plus the strongest safe neighbors for each UE (`K=8`),
2. shares one Q-network across all UEs,
3. outputs one Q-value per UE–candidate edge, and
4. executes all per-UE choices together.

The neural network therefore produces only `100 × 8 = 800` scores per decision, while still using global cell/load context through the bipartite GNN.

## State, action, and reward

### Graph

- **UE nodes:** position, velocity, traffic demand, time since handover, serving SINR.
- **Cell nodes:** position, resource-load proxy, activity, and connected-UE count.
- **UE→cell edges:** RSRP, SINR, distance, serving flag, target load, and RSRP difference.
- **Sparse edges:** only the serving cell and top candidate cells are stored.

### Action

For every UE, action `0` means stay; actions `1..K-1` select a candidate neighbor. Unsafe candidates are masked. This is a direct association/handover controller, not a CIO controller; call it that in the paper.

### Per-UE reward

`0.60 × demand satisfaction + 0.15 × Jain fairness − 0.05 × handover − 0.25 × ping-pong − 0.60 × RLF − 0.10 × overload`

Do not tune weights on the test set. Select them on validation seeds, then freeze them.

## Toolchain to use

### Stage A — write and debug the algorithm (this repository)

- Python 3.10+
- PyTorch (no PyTorch Geometric installation is required; sparse bipartite attention is implemented directly)
- NumPy/Pandas
- Google Colab or a local NVIDIA GPU

### Stage B — realistic mobility

Replace the built-in random mobility with **Eclipse SUMO** routes through Python TraCI/libsumo. SUMO is designed for microscopic urban mobility and its Python interface can supply UE positions at every simulation step: <https://sumo.dlr.de/docs/TraCI/index.html>

### Stage C — paper-grade network validation

Port the learned policy to **ns-3 with LTE LENA or 5G-LENA**. 5G-LENA is an open-source end-to-end NR module with PHY/MAC and 3GPP channel support: <https://5g-lena.cttc.es/>. The current Python environment is intentionally transparent and fast, but it is **not** a full protocol-stack or complete TR 38.901 implementation.

The graph-RL connection-management idea is grounded in Orhan et al.: <https://arxiv.org/abs/2110.07525>. That work is user association; your evaluation must explicitly demonstrate mobile handover behavior, TTT effects, ping-pong, and failure behavior.

## Installation

### Google Colab

Colab normally includes PyTorch. Upload and unzip the project, then run:

```bash
%cd /content/gnn_ddqn_handover
!pip install -q -r requirements.txt
!python smoke_test.py
!python train.py --config configs/smoke.yaml --out runs/smoke --steps 400
```

### Local Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py
```

## Train the requested 100-cell/100-UE model

```bash
python train.py \
  --config configs/dense_urban_100x100.yaml \
  --out runs/dense_urban_100x100 \
  --device auto
```

Outputs:

- `train_episodes.csv` — training curves
- `validation.csv` — deterministic validation episodes
- `checkpoint_best.pt` — best validation checkpoint
- `checkpoint_last.pt` — latest checkpoint

Resume a stage with:

```bash
python train.py --config configs/dense_urban_100x100.yaml \
  --out runs/dense_urban_100x100_stage2 \
  --resume runs/dense_urban_100x100/checkpoint_best.pt
```

When changing graph size, resume weights but use a new run directory; the replay buffer should start empty.

## Recommended training sequence

1. **Environment smoke test:** 9 cells/24 UEs, 400 steps.
2. **Algorithm debug:** 25 cells/100 UEs until reward and TD loss are stable.
3. **Scale training:** 50 cells/200 UEs; resume the network, clear replay.
4. **Target training:** 100 cells/100 UEs using the requested config.
5. **Load stress:** evaluate or fine-tune on 100 cells/300 UEs.
6. **Generalization:** new maps, unseen mobility seeds, shadowing levels, demand, cell failures, and speed mixtures.
7. **Paper validation:** repeat the frozen policy in SUMO + ns-3/5G-LENA and/or on field traces.

Do not call the first rising reward curve “optimized.” Save the checkpoint using validation seeds only.

## Baseline evaluation

The evaluator uses the **same seed/map/traffic realization for every policy**:

```bash
python evaluate.py \
  --config configs/dense_urban_100x100.yaml \
  --checkpoint runs/dense_urban_100x100/checkpoint_best.pt \
  --out evaluation/100x100 \
  --seeds 30
```

Included baselines:

- Stay with serving cell
- strongest RSRP
- Event-A3-style hysteresis + TTT
- load-aware heuristic
- learned GNN-DDQN

It writes per-seed values and mean, standard deviation, and approximate 95% confidence intervals. For the paper, additionally run paired bootstrap intervals or a Wilcoxon signed-rank test against A3-TTT.

## Minimum experiment matrix for the paper

Use **held-out seeds and maps**, not adjacent timesteps from training.

| Dimension | Values |
|---|---|
| UEs | 100, 300, 500 |
| Cells | 100 |
| Mobility | pedestrian, mixed urban, vehicular |
| Shadowing σ | 4, 7, 10 dB |
| Demand | light, medium, heavy |
| Cell availability | normal, 5% failed |
| Seeds | at least 30 paired test seeds |

Primary metrics: mean and 5th-percentile UE throughput, demand satisfaction, Jain fairness, handovers/UE/min, ping-pong fraction, RLF/UE/min, overload fraction, inference latency, and parameter count.

## Required ablations

1. GNN-DDQN vs MLP-DDQN (tests graph value).
2. Double+Dueling vs plain DQN.
3. Prioritized vs uniform replay.
4. Safety mask on vs off.
5. Reward without ping-pong term.
6. `K ∈ {4, 8, 12}` candidate sensitivity.

## Important limitations before publication

- The provided path loss, scheduler, RLF, and A3 behavior are explicit **research proxies**.
- The built-in A3 timer uses the simulator step size; final TTT should be validated in a protocol-stack simulator.
- One UE per cell is not a strong load-balancing case.
- Never report invented improvements. All percentages must come from the frozen-checkpoint evaluator on held-out seeds.
- Use “first” or “novel” only after a systematic literature search proves it.
