"""
Self-Pruning Neural Network on CIFAR-10


Architecture:
- Custom PrunableLinear layers with learnable sigmoid gates
- Sparsity regularization via L1 penalty on gates
- Evaluated across multiple lambda values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# Part 1: Custom PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A fully connected layer augmented with learnable gate parameters.

    Each weight w_ij has a corresponding gate score g_ij (also a learned
    parameter). During the forward pass, gates are squashed through a
    Sigmoid to produce values in (0, 1). The effective weight becomes:
        pruned_weight = weight * sigmoid(gate_score)

    Gradients flow through both `weight` and `gate_scores` via autograd,
    so the optimizer can drive gates toward 0 (pruning) or 1 (keeping).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias — same init as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores: same shape as weight; initialized near 0 so gates
        # start at ~0.5 (sigmoid(0)), giving the network a neutral start.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming uniform init for weights (good for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert gate scores → gates ∈ (0, 1) via Sigmoid
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Element-wise gate application — gradients flow through both tensors
        pruned_weights = self.weight * gates             # shape: (out, in)

        # Standard affine transform: x @ W^T + b
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached) for analysis."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of gate values — encourages gates toward 0."""
        return torch.sigmoid(self.gate_scores).abs().sum()


# ─────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 (32×32×3 → 10 classes).
    All linear layers use PrunableLinear.
    """

    def __init__(self):
        super().__init__()
        # CIFAR-10 images: 3×32×32 = 3072 input features
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512,  256)
        self.fc3 = PrunableLinear(256,  128)
        self.fc4 = PrunableLinear(128,   10)

        self.prunable_layers = [self.fc1, self.fc2, self.fc3, self.fc4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)          # flatten: (B, 3072)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)                     # logits (no softmax; CE handles it)
        return x

    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of L1 gate penalties across all prunable layers."""
        return sum(layer.sparsity_loss() for layer in self.prunable_layers)

    def compute_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Fraction of weights whose gate value is below `threshold`.
        A high value means successful pruning.
        """
        all_gates = torch.cat(
            [layer.get_gates().view(-1) for layer in self.prunable_layers]
        )
        pruned = (all_gates < threshold).float().sum()
        return (pruned / all_gates.numel()).item()

    def all_gate_values(self) -> np.ndarray:
        """Collect all gate values as a flat numpy array (for plotting)."""
        all_gates = torch.cat(
            [layer.get_gates().view(-1) for layer in self.prunable_layers]
        )
        return all_gates.cpu().numpy()


# ─────────────────────────────────────────────
# Part 3: Data Loading
# ─────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ─────────────────────────────────────────────
# Part 3: Training Loop
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, lambda_sparse, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        # Total Loss = Cross-Entropy + λ * L1(gates)
        clf_loss     = F.cross_entropy(logits, labels)
        sparsity_loss = model.total_sparsity_loss()
        loss         = clf_loss + lambda_sparse * sparsity_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total


# ─────────────────────────────────────────────
# Main Experiment: sweep over λ values
# ─────────────────────────────────────────────

def run_experiment(lambda_sparse: float, epochs: int, device):
    print(f"\n{'='*55}")
    print(f"  Training with λ = {lambda_sparse}")
    print(f"{'='*55}")

    train_loader, test_loader = get_cifar10_loaders()
    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     lambda_sparse, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            acc      = evaluate(model, test_loader, device)
            sparsity = model.compute_sparsity()
            print(f"  Epoch {epoch:3d} | Loss {train_loss:.4f} | "
                  f"Acc {acc*100:.2f}% | Sparsity {sparsity*100:.1f}%")

    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = model.compute_sparsity()
    gate_values    = model.all_gate_values()

    print(f"\n  ✓ Final  Accuracy : {final_acc*100:.2f}%")
    print(f"  ✓ Final  Sparsity : {final_sparsity*100:.1f}%")

    return final_acc, final_sparsity, gate_values


def plot_gate_distributions(results: dict, best_lambda: float):
    """
    Plot gate value distributions for each λ.
    A successful pruning shows a spike near 0 and a cluster near 1.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4),
                             sharey=False)
    if len(results) == 1:
        axes = [axes]

    for ax, (lam, (acc, sparsity, gates)) in zip(axes, results.items()):
        color = "#2196F3" if lam != best_lambda else "#FF5722"
        ax.hist(gates, bins=80, color=color, edgecolor="none", alpha=0.85)
        ax.set_title(f"λ = {lam}\nAcc={acc*100:.1f}%  Sparsity={sparsity*100:.1f}%",
                     fontsize=11)
        ax.set_xlabel("Gate Value")
        ax.set_ylabel("Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Gate Value Distribution after Self-Pruning Training",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("gate_distributions.png", dpi=150, bbox_inches="tight")
    print("\n  📊 Plot saved → gate_distributions.png")
    plt.show()


if __name__ == "__main__":
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS  = 20          # increase to 30–40 for better accuracy
    LAMBDAS = [1e-4, 1e-3, 5e-3]   # low, medium, high sparsity pressure

    results = {}
    for lam in LAMBDAS:
        acc, sparsity, gates = run_experiment(lam, EPOCHS, device)
        results[lam] = (acc, sparsity, gates)

    # ── Summary Table ───────────────────────────────────────────────────
    print("\n" + "="*52)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>15}")
    print("="*52)
    for lam, (acc, sparsity, _) in results.items():
        print(f"  {lam:<12} {acc*100:>14.2f}%  {sparsity*100:>13.1f}%")
    print("="*52)

    # Best model = highest accuracy
    best_lambda = max(results, key=lambda l: results[l][0])
    print(f"\n  Best λ (by accuracy): {best_lambda}")

    # ── Plot ────────────────────────────────────────────────────────────
    plot_gate_distributions(results, best_lambda)
