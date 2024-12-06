import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Tuple
from copy import deepcopy

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.width = hidden_dims[0] 

        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))

        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        for i, layer in enumerate(self.layers):
            # Scale weights by 1/√width for hidden layers
            if i > 0 and i < len(self.layers) - 1:
                std = 1.0 / np.sqrt(self.width)
            else:
                std = 1.0 / np.sqrt(layer.weight.shape[1])
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.tanh(layer(x))
        return self.layers[-1](x)

def generate_unit_circle_data(n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate points on a unit circle with their corresponding labels."""
    theta = torch.linspace(0, 2*np.pi, n_samples)
    X = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    y = torch.sin(2 * theta).reshape(-1, 1)
    return X, y

def compute_ntk(model: nn.Module, x1: torch.Tensor, x2: torch.Tensor = None) -> torch.Tensor:
    """Compute the Neural Tangent Kernel between x1 and x2."""
    if x2 is None:
        x2 = x1

    # Get gradients for x1
    y1 = model(x1)
    grad_list1 = []
    for i in range(y1.shape[0]):
        model.zero_grad()
        y1[i].backward(retain_graph=True)
        grad1 = torch.cat([p.grad.flatten() for p in model.parameters()])
        grad_list1.append(grad1)

    # Get gradients for x2
    y2 = model(x2)
    grad_list2 = []
    for i in range(y2.shape[0]):
        model.zero_grad()
        y2[i].backward(retain_graph=True)
        grad2 = torch.cat([p.grad.flatten() for p in model.parameters()])
        grad_list2.append(grad2)

    # Compute kernel matrix
    K = torch.zeros((x1.shape[0], x2.shape[0]))
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            K[i,j] = torch.dot(grad_list1[i], grad_list2[j])

    return K

def compute_ntk_difference(ntk1: torch.Tensor, ntk2: torch.Tensor) -> float:
    """Compute normalized Frobenius norm difference between two NTK matrices."""
    return torch.norm(ntk1 - ntk2, p='fro').item() / torch.norm(ntk1, p='fro').item()

class PolyakHeavyBall:
    """Implementation of Polyak's Heavy Ball optimizer."""
    def __init__(self, params, lr: float, momentum: float = 0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * param.grad
                param.data.add_(self.velocity[i])

def train_and_track_ntk(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_steps: int = 300,
    track_frequency: int = 10,
    momentum: float = 0.9
) -> Tuple[List[float], List[float], List[torch.Tensor]]:
    """Train the model with Polyak HB and track NTK evolution."""
    learning_rate = 0.1 / model.width
    optimizer = PolyakHeavyBall(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.MSELoss()

    # Initial NTK
    initial_ntk = compute_ntk(model, X)
    ntk_differences = [0.0]  
    losses = [criterion(model(X), y).item()]
    ntk_matrices = [initial_ntk]  

    for step in range(n_steps):
        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track NTK evolution at specified frequency
        if (step + 1) % track_frequency == 0:
            current_ntk = compute_ntk(model, X)
            diff = compute_ntk_difference(initial_ntk, current_ntk)
            ntk_differences.append(diff)
            losses.append(loss.item())
            ntk_matrices.append(current_ntk)

            if (step + 1) % 50 == 0:
                print(f"Step {step + 1}, Loss: {loss.item():.6f}, NTK Difference: {diff:.6f}")

    return losses, ntk_differences, ntk_matrices

def analyze_ntk_convergence():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generate data
    X, y = generate_unit_circle_data(50)  # Reduced sample size for faster computation

    # Network widths to analyze
    widths = [100, 500, 2000, 8000]
    n_steps = 300
    track_frequency = 10

    # Store results
    all_losses = {}
    all_ntk_diffs = {}
    all_ntk_matrices = {}

    # Train networks of different widths
    for width in widths:
        print(f"\nTraining network with width {width} using Polyak Heavy Ball")
        model = NeuralNetwork(
            input_dim=2,
            hidden_dims=[width, width],
            output_dim=1
        )

        losses, ntk_diffs, ntk_matrices = train_and_track_ntk(
            model, X, y,
            n_steps=n_steps,
            track_frequency=track_frequency
        )

        all_losses[width] = losses
        all_ntk_diffs[width] = ntk_diffs
        all_ntk_matrices[width] = ntk_matrices

    # Create visualization
    plt.figure(figsize=(15, 12))

    # Plot NTK differences
    plt.subplot(221)
    for width in widths:
        steps = np.arange(0, n_steps + 1, track_frequency)[:len(all_ntk_diffs[width])]
        plt.plot(steps, all_ntk_diffs[width], label=f'Width {width}')

    plt.title('NTK Evolution with Polyak Heavy Ball')
    plt.xlabel('Training Steps')
    plt.ylabel('Normalized NTK Difference')
    plt.legend()
    plt.grid(True)

    # Plot losses
    plt.subplot(222)
    for width in widths:
        steps = np.arange(0, n_steps + 1, track_frequency)[:len(all_losses[width])]
        plt.plot(steps, all_losses[width], label=f'Width {width}')

    plt.title('Loss Evolution with Polyak Heavy Ball')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot NTK heatmaps for smallest and largest width
    plt.subplot(223)
    plt.imshow(all_ntk_matrices[widths[0]][-1].detach().numpy())
    plt.title(f'Final NTK Matrix (Width {widths[0]})')
    plt.colorbar()

    plt.subplot(224)
    plt.imshow(all_ntk_matrices[widths[-1]][-1].detach().numpy())
    plt.title(f'Final NTK Matrix (Width {widths[-1]})')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_ntk_convergence()