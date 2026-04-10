# FairFlow — Two-Objective RL for Throughput-Fairness Tradeoff in Traffic Control

> A deep reinforcement learning system that optimizes traffic signal control at 4-way intersections by simultaneously maximizing vehicle throughput **and** ensuring equitable treatment across vehicle types.

---

## Team

| Name | Registration No. |
|------|-----------------|
| Aditya Rajeev Nair | 23BAI0036 |
| Agnik Patra | 23BAI001 |
| Anasmita Bhattacharya | 23BAI0028 |
| Paulami Sahu | 23BAI0020 |

---

## Overview

Traffic signal optimization typically prioritizes throughput, inadvertently creating inequitable waiting conditions across different vehicle categories. **FairFlow** tackles this as a two-objective RL problem — jointly optimizing for efficiency and fairness — using an attention-enhanced Dueling DQN architecture.

The core contribution is demonstrating the **throughput-fairness Pareto tradeoff** and showing that a dual-objective agent can navigate it meaningfully, outperforming a throughput-only baseline in equity metrics while maintaining competitive throughput.

---

## Architecture

### Agent: Attention-Enhanced Dueling Deep Q-Network

```
State (36-dim) ──► Vehicle Type Embeddings (16-dim)
                ──► SpatialTemporalAttention (4 heads)
                ──► DuelingNetwork (Value + Advantage streams)
                ──► Q-values → Action {keep phase | switch phase}
                └──► FairnessPredictor (auxiliary head)
```

**Key components:**

| Module | Description |
|--------|-------------|
| `SpatialTemporalAttention` | 4-head attention over traffic directions and time to capture complex spatio-temporal patterns |
| `DuelingNetwork` | Separates value V(s) and advantage A(s,a) for improved learning stability |
| `FairnessPredictor` | Auxiliary network predicting fairness metrics as a side task, guiding representation learning |
| Experience Replay | Prioritized buffer with capacity 5 000 transitions |
| Epsilon-Greedy | Exploration decay from ε=1.0 → 0.01 over training |

---

## Environment

**`FairFlowTrafficEnvironment`** — custom 4-way intersection simulator built on [Gymnasium](https://gymnasium.farama.org/).

### Vehicle Types

| Type | Priority | Size | Service Rate | Spawn Probability |
|------|----------|------|--------------|-------------------|
| PUBLIC_TRANSIT | 0.8 | 0.8 | 0.5 veh/step | 0.08 |
| PRIVATE_CAR | 0.5 | 0.5 | 1.0 veh/step | 0.25 |
| TRUCK | 0.3 | 0.9 | 0.4 veh/step | 0.12 |
| MOTORCYCLE | 0.4 | 0.2 | 1.5 veh/step | 0.15 |

### State Space (36 dimensions)

| Feature Group | Dims | Description |
|---------------|------|-------------|
| Queue lengths | 16 | Per direction × vehicle type |
| Average wait times | 4 | Per direction |
| Phase info | 2 | Current phase + duration |
| Fairness metrics | 4 | Per vehicle type |
| Time encoding | 2 | Hour of day + day of week |
| Arrival history | 8 | 8-step rolling window |

### Action Space

- **0** — Keep current traffic phase  
- **1** — Switch to next traffic phase

### Key Parameters

| Parameter | Value |
|-----------|-------|
| Episode length | 200 timesteps |
| Max queue per direction | 20 vehicles |
| Minimum phase duration | 3 timesteps |
| Training episodes | 300–500 |

---

## Reward Function

The reward combines two objectives:

```
R = α · Throughput_reward + β · Fairness_reward
```

Fairness is measured via **Jain's Fairness Index**:

```
J(x) = (Σxᵢ)² / (n · Σxᵢ²)     ∈ [0, 1]
```

Higher J → more equitable wait time distribution across vehicle types.

---

## Baseline

**`ThroughputOnlyAgent`** — identical architecture but optimizes throughput alone (β=0). Used to quantify the cost of ignoring fairness and to plot the Pareto frontier.

---

## Hyperparameters

| Hyperparameter | Value |
|---------------|-------|
| Learning rate | 0.0005 |
| Discount factor γ | 0.99 |
| Batch size | 64 |
| Replay memory | 5 000 |
| Hidden dimensions | 128–256 |
| Attention heads | 4 |
| Vehicle type embedding dim | 16 |

---

## Metrics & Evaluation

| Metric | Description |
|--------|-------------|
| **Jain's Fairness Index** | Equity across vehicle types (↑ better, max 1.0) |
| **Gini Coefficient** | Wait time inequality (↓ better, min 0.0) |
| **Throughput** | Total vehicles served per episode |
| **Average Reward** | Cumulative two-objective return |
| **Pareto Front** | 2D tradeoff curve: throughput vs. fairness |

---

## Results & Visualizations

The notebook generates the following plots:

- Learning curves with confidence intervals
- Fairness evolution (Jain's Index) over training
- Vehicle-type throughput breakdown
- Wait time distribution by vehicle type
- Reward component distribution
- Training loss curves
- Gini coefficient over episodes
- **Pareto frontier** — FairFlow vs. ThroughputOnly baseline
- Box plots and grouped bar charts for wait time analysis

---

## Dependencies

```
torch
gymnasium
numpy
pandas
scipy
matplotlib
seaborn
```

> The notebook was developed on **Google Colab** with an **NVIDIA T4 GPU**. PyTorch with CUDA acceleration is recommended for training.

---

## Usage

1. Open `Fairflow.ipynb` in Google Colab or a local Jupyter environment.
2. Install dependencies (all standard; pre-installed on Colab).
3. Run all cells sequentially — training, evaluation, and all visualizations are self-contained in the notebook.
4. Trained model weights are saved to `final_fairflow_traffic_model.pth` at the end of training.

---

## Project Structure

```
FairFlow_RLProject/
├── Fairflow.ipynb          # Main notebook (all code, training, results)
├── README.md               # This file
└── final_fairflow_traffic_model.pth   # Saved model weights (generated after training)
```

---

## Key Takeaways

- A purely throughput-maximizing agent systematically disadvantages lower-priority vehicle types (trucks, motorcycles).
- The dual-objective FairFlow agent achieves **near-Pareto-optimal** behavior, demonstrating that significant fairness gains are achievable with minimal throughput cost.
- Attention mechanisms meaningfully help the agent learn spatio-temporal traffic patterns compared to flat MLP baselines.
- The Jain's Index + Gini Coefficient combination provides a robust, complementary view of distributional fairness.
