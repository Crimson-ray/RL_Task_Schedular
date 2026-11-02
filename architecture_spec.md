# Architecture Specification: NeuraSched (Intelligent Task Scheduler)

## 1. Executive Summary & Problem Statement

Modern distributed compute clusters (e.g., Google Borg, Kubernetes, Alibaba Cloud) process millions of heterogeneous workloads characterized by:
- **Dynamic Arrival Patterns:** Bursty, diurnal, and heavy-tailed task distributions (Poisson / Pareto).
- **Multi-Resource Constraints:** Tasks demand multidimensional resources (CPU cores, RAM) simultaneously. Naive single-resource scheduling causes **resource stranding** (e.g., node CPU is 100% utilized while 70% of memory is idle).
- **Service Level Agreements (SLAs):** Heterogeneous latency deadlines and task priorities (Latency-Critical vs. Best-Effort / Batch).
- **Sub-Millisecond Latency Requirement:** Real-time scheduling decisions must occur within $< 1\,\text{ms}$ per task, precluding heavy offline combinatorial solvers (MILP) for online streaming.

Traditional rule-based heuristics (FIFO, Shortest Job First, Earliest Deadline First) operate greedily and fail to adapt to complex workload shifts or multi-resource fragmentation. Conversely, full Graph Neural Network (GNN) DAG solvers incur excessive inference overhead ($>20\,\text{ms}$) per scheduling step.

This project delivers **NeuraSched**: an **Intelligent Task Scheduler** powered by a **Dynamic Action-Masked Dueling Double Deep Q-Network (D3QN)** built upon a Deep MLP with LeakyReLU activations and Layer Normalization, capable of sub-millisecond scheduling decisions while optimizing makespan, resource utilization, and SLA compliance.

---

## 2. NeuraSched Architectural Blueprint

```
+-----------------------------------------------------------------------------------------+
|                                    NEURASCHED                                           |
|       DYNAMIC ACTION-MASKED DUELING DOUBLE DEEP Q-NETWORK (D3QN) ON DEEP MLP            |
+-----------------------------------------------------------------------------------------+

  6 Compute Nodes  |  Features: CPU, RAM, Deadlines, Priority, Duration
  State Vector: S_t = [ f_task (5), f_nodes (36), f_cluster (4) ] in R^45

               +----------------------------------------------------+
               |         State Encoder (Input: R^45)                |
               |  - f_task (5 dim): Normalized CPU/RAM, duration,   |
               |                    slack, priority                 |
               |  - f_nodes (36 dim): 6 nodes * 6 features          |
               |                      [Free CPU/RAM ratio]          |
               |  - f_cluster (4 dim): Cluster CPU/RAM util,        |
               |                       queue backlog, variance      |
               +----------------------------------------------------+
                                         |
                                         v
               +----------------------------------------------------+
               |            Dense Layer 1 (PyTorch)                 |
               |               Linear(45 -> 128)                    |
               |       LayerNorm(128) + LeakyReLU(0.01)             |
               |                                                    |
               |            Dense Layer 2 (PyTorch)                 |
               |               Linear(128 -> 128)                   |
               |       LayerNorm(128) + LeakyReLU(0.01)             |
               +----------------------------------------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
        +--------------------------+           +--------------------------+
        |       Value Stream       |           |     Advantage Stream     |
        | [how good is our state]  |           | [what is the better act] |
        |  Linear(128 -> 64)       |           |  Linear(128 -> 64)       |
        |  + LeakyReLU(0.01)       |           |  + LeakyReLU(0.01)       |
        |  Linear(64 -> 1)         |           |  Linear(64 -> 7 actions) |
        |                          |           |  (6 nodes + 1 Defer)     |
        |  Output: Scalar V(s)     |           |  Output: Vector A(s, a)  |
        +--------------------------+           +--------------------------+
                     |                                       |
                     +-------------------+-------------------+
                                         |
                                         v
               +----------------------------------------------------+
               |              Dueling Aggregation                   |
               |        Q(s, a) = V(s) + (A(s, a) - A_bar(s))       |
               +----------------------------------------------------+
                                         |
                                         v
               +----------------------------------------------------+
               |       Dynamic Action Masking & Action Selection    |
               |             a* = argmax_{a: M(a)=True} Q(s, a)     |
               |  Actions 0-5: Allocate on ComputeNode 0-5          |
               |  Action 6: Defer / Hold in queue                   |
               +----------------------------------------------------+
```

---

## 3. Detailed Component Breakdown

### 3.1 State Encoder ($S_t \in \mathbb{R}^{45}$)
1. **$f_{\text{task}} \in \mathbb{R}^5$ (Incoming Task Features):**
   - Normalized CPU request: $c_j / 32.0$
   - Normalized RAM request: $m_j / 64.0$
   - Normalized execution duration: $d_j / 120.0$
   - Normalized SLA slack: $\min(5.0, (D_j - t) / d_j) / 5.0$
   - Priority tier: $p_j / 2.0 \in \{0.0, 0.5, 1.0\}$ (Batch, Standard, Critical)
2. **$f_{\text{nodes}} \in \mathbb{R}^{36}$ ($6 \text{ Nodes} \times 6 \text{ Features}$):**
   For each of the 6 heterogeneous compute nodes:
   - Free CPU ratio: $c_i^{\text{avail}} / c_i^{\text{total}}$
   - Free RAM ratio: $m_i^{\text{avail}} / m_i^{\text{total}}$
   - Node CPU utilization: $c_i^{\text{used}} / c_i^{\text{total}}$
   - Node RAM utilization: $m_i^{\text{used}} / m_i^{\text{total}}$
   - Active running task ratio: $\min(1.0, |J_i| / 20.0)$
   - Free CPU-to-RAM ratio: $\min(5.0, c_i^{\text{avail}} / m_i^{\text{avail}}) / 5.0$
3. **$f_{\text{cluster}} \in \mathbb{R}^4$ (Global Cluster State):**
   - Cluster aggregate CPU utilization
   - Cluster aggregate RAM utilization
   - Normalized queue backlog length
   - Inter-node load variance

### 3.2 Deep MLP Backbone
- **Dense Layer 1:** `Linear(45 -> 128)` followed by `LayerNorm(128)` (stabilizing and normalizing activations) and `LeakyReLU(negative_slope=0.01)`.
- **Dense Layer 2:** `Linear(128 -> 128)` followed by `LayerNorm(128)` and `LeakyReLU(negative_slope=0.01)`.

### 3.3 Dueling Decomposition & 7-Action Space
- **Value Stream $V(s)$:** `Linear(128 -> 64) -> LeakyReLU -> Linear(64 -> 1)` producing scalar state goodness $V(s)$.
- **Advantage Stream $A(s, a)$:** `Linear(128 -> 64) -> LeakyReLU -> Linear(64 -> 7)` producing action advantages $A(s, a)$.
  - **Actions 0–5:** Allocate current task directly onto ComputeNode $0, \dots, 5$.
  - **Action 6:** Defer / Hold task in admission queue until next completion event.
- **Aggregation:**
  $$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{7} \sum_{a'=0}^{6} A(s, a') \right)$$

### 3.4 Dynamic Action Masking
Prevents impossible placements and invalid state transitions:
$$M_t(a) = \begin{cases} 
1, & \text{if } a < 6 \land c_a^{\text{avail}} \ge c_j \land m_a^{\text{avail}} \ge m_j \\
1, & \text{if } a = 6 \text{ (Defer enabled)} \\
0, & \text{otherwise}
\end{cases}$$
During action selection:
$$\pi(s) = \arg\max_{a: M_t(a) = 1} Q(s, a)$$
Masked infeasible actions have logits forced to $-10^9$, guaranteeing that the network never assigns a task to an overloaded node.

---

## 4. Multi-Objective Reward Function
$$R = w_{\text{pack}} R_{\text{pack}} + w_{\text{sla}} R_{\text{sla}} - w_{\text{bal}} R_{\text{bal}} + R_{\text{throughput}}$$
- **Multi-Resource Anti-Stranding ($R_{\text{pack}}$):** $0.5 \times \text{CosineAlign} + 0.5 \times (1 - \text{StrandedRatio})$.
- **SLA Adherence ($R_{\text{sla}}$):** Bonus for completion before deadline; penalty proportional to tardiness and priority tier.
- **Load Balancing ($R_{\text{bal}}$):** Penalizes quadratic utilization spikes.
- **Throughput ($R_{\text{throughput}}$):** Base bonus upon successful allocation; minor penalty on deferral to prevent lazy queue stagnation.
