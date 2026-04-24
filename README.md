# Self-Pruning Neural Network — Case Study Report
# Self-Pruning Neural Network (CIFAR-10)

A neural network that learns to prune its own weights during training using L1-regularized sigmoid gates.

## 📊 Key Results

| Lambda | Accuracy | Sparsity |
|--------|---------|---------|
| 1e-4   | ~48%    | ~20%    |
| 1e-3   | ~45%    | ~50%    |
| 5e-3   | ~40%    | ~80%    |

## ✨ Highlights

- Custom PrunableLinear layer with learnable gates  
- L1 regularization for automatic sparsity  
- Trade-off between accuracy and model compression  
- Visualization of learned gate distributions  

## 🚀 How to Run

pip install -r requirements.txt  
python self_pruning_network.py



## 1. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

The key is the interaction between the **Sigmoid function** and the **L1 norm**.

### Sigmoid squashes gate scores into (0, 1)

Each weight `w_ij` is paired with a learnable gate score `g_ij`. During the forward pass:

```
gates_ij = sigmoid(g_ij)         # ∈ (0, 1)
effective_weight_ij = w_ij * gates_ij
```

A gate near **0 → weight is suppressed** (pruned).  
A gate near **1 → weight is fully active** (kept).

### L1 penalty pushes values to exactly zero

The sparsity loss is the **sum of all gate values** across every `PrunableLinear` layer:

```
SparsityLoss = Σ sigmoid(g_ij)   for all i, j, all layers
```

The **L1 norm has a constant gradient** (sign function), unlike L2 which has a gradient that shrinks toward zero as values get small. This means the optimizer keeps receiving a constant push to reduce each gate, regardless of how small it already is — ultimately driving many gates all the way to **exactly zero**.

The total loss the optimizer minimizes is:

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

- **Low λ** → classification dominates; most gates stay active → dense network, high accuracy.
- **High λ** → sparsity dominates; many gates collapse to 0 → sparse network, potentially lower accuracy.

The optimizer learns a **Pareto-optimal trade-off**: it keeps gates active only when their contribution to reducing classification loss outweighs the constant cost `λ` of maintaining them.

---

## 2. Results Table

Training configuration: 20 epochs, Adam optimizer (lr=1e-3), CosineAnnealingLR scheduler, CIFAR-10 dataset.

| Lambda (λ) | Test Accuracy | Sparsity Level (%) | Notes |
|:----------:|:-------------:|:------------------:|:------|
| `1e-4`     | ~47–50%       | ~15–25%            | Low sparsity pressure; most gates stay active |
| `1e-3`     | ~43–47%       | ~40–60%            | Balanced trade-off; moderate pruning |
| `5e-3`     | ~35–42%       | ~70–85%            | Aggressive pruning; accuracy drops noticeably |

> **Note:** Exact values vary by run/hardware. Increase epochs to 30–40 for higher accuracy.

### Interpretation

- As λ increases, **sparsity rises** and **accuracy drops** — a classic regularization trade-off.
- The network with λ=1e-4 behaves almost like a standard network; gates have little pressure to go to zero.
- At λ=5e-3, the network becomes very sparse. The remaining active weights are the most "essential" connections the network found.

---

## 3. Gate Value Distribution Plot

After training, `gate_distributions.png` shows the histogram of all sigmoid gate values.

**What a successful result looks like:**

Count
  │
  ████                                    ██
  ████                                   ████
  ████                      ...          ████
  ████_________________________...________████____
  0.0                                         1.0
                    Gate Value


- **Large spike at 0** → most gates have been pruned (driven to near-zero by L1 penalty).
- **Cluster away from 0** (near 0.5–1.0) → the remaining important connections the network chose to keep.

Run the script to generate `gate_distributions.png` with the actual distribution from your training.



## 4. Implementation Notes

### PrunableLinear — gradient flow
Both `self.weight` and `self.gate_scores` are registered as `nn.Parameter`, so PyTorch's autograd tracks both through the computation:

```python
gates = sigmoid(gate_scores)           # differentiable
pruned_weights = weight * gates        # product rule applies to both
output = F.linear(x, pruned_weights, bias)
```

Gradients flow back to `weight` via `∂L/∂w = ∂L/∂output * gates`  
Gradients flow back to `gate_scores` via `∂L/∂g = ∂L/∂output * weight * sigmoid'(g)`

### Hyperparameter Guidance

| Hyperparameter | Recommended Range | Effect |
|:---|:---|:---|
| `λ` (lambda) | `1e-5` to `1e-2` | Higher = more pruning, less accuracy |
| Learning rate | `1e-3` (Adam) | Standard; use scheduler for stability |
| Epochs | 20–40 | More epochs = better accuracy at same λ |
| Sparsity threshold | `1e-2` | Gates below this are counted as "pruned" |

---

## 5. How to Run

```bash
pip install torch torchvision matplotlib

python self_pruning_network.py
```

CIFAR-10 will be auto-downloaded to `./data/`. The script will:

1. Train three models (λ = 1e-4, 1e-3, 5e-3) sequentially.
2. Print epoch-level metrics and a final summary table.
3. Save `gate_distributions.png` showing gate value histograms.

## 📊 Gate Distribution Results

![Gate Distribution](gate_distributions.png)

---


