import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, beta=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.width = hidden_dims[0]
        self.beta = beta  # Parameter to control bias influence

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self._initialize_weights()
        self.to(device)

    def _initialize_weights(self):
        """Initialize weights with N(0,1) as per paper"""
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            nn.init.normal_(layer.bias, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing α̃(ℓ+1)(x;θ) = 1/√nℓ W(ℓ)α(ℓ)(x;θ) + βb(ℓ)"""
        for i, layer in enumerate(self.layers[:-1]):
            # Separate weight and bias terms
            w_out = layer.weight @ (x.T if x.dim() > 1 else x.unsqueeze(1))
            w_out = w_out.T / np.sqrt(layer.weight.shape[1])  # 1/√nℓ * W(ℓ)x
            b_out = self.beta * layer.bias  # βb(ℓ)
            x = F.relu(w_out + b_out.unsqueeze(0) if x.dim() > 1 else b_out)
        
        # Final layer
        layer = self.layers[-1]
        w_out = layer.weight @ (x.T if x.dim() > 1 else x.unsqueeze(1))
        w_out = w_out.T / np.sqrt(layer.weight.shape[1])
        b_out = self.beta * layer.bias
        return w_out + b_out.unsqueeze(0) if x.dim() > 1 else b_out

def compute_ntk_slice(model: nn.Module, x0: torch.Tensor, x_points: torch.Tensor) -> torch.Tensor:
    """Compute NTK between fixed point x0 and all points in x_points."""
    model.eval()  # Set to eval mode
    
    # Get gradients for x0
    y0 = model(x0)
    y0.requires_grad_(True)  # Enable gradient computation
    model.zero_grad()
    y0.backward(torch.ones_like(y0), retain_graph=True)
    grad0 = torch.cat([p.grad.flatten() for p in model.parameters()])
    
    # Get gradients for x_points
    ntk_values = []
    y_points = model(x_points)
    y_points.requires_grad_(True)  # Enable gradient computation
    
    for i in range(y_points.shape[0]):
        model.zero_grad()
        y_points[i].backward(torch.ones_like(y_points[i]), retain_graph=True)
        grad_i = torch.cat([p.grad.flatten() for p in model.parameters()])
        ntk_values.append(torch.dot(grad0, grad_i).item())
    
    model.train()  # Set back to train mode
    return torch.tensor(ntk_values, device=device)

def plot_figure1(widths=[500, 10000], n_points=100, n_inits=10, train_network=None, save_dir=None):
    """Generate all figures with multiple initializations."""
    gamma = torch.linspace(-3, 3, n_points, device=device)
    X = torch.stack([torch.cos(gamma), torch.sin(gamma)], dim=1)
    x0 = torch.tensor([[1.0, 0.0]], device=device)
    y = 0.5 * torch.sin(2 * gamma).reshape(-1, 1)

    results = {str(width): {
        'ntk_init': [], 'ntk_final': [], 'losses': []
    } for width in widths}
    
    for width in widths:
        print(f"\nProcessing width {width} with {n_inits} initializations...")
        
        for init in range(n_inits):
            print(f"  Initialization {init + 1}/{n_inits}")
            torch.manual_seed(42 + init)
            
            model = NeuralNetwork(
                input_dim=2,
                hidden_dims=[width] * 3,
                output_dim=1
            )
            
            # Collect results
            ntk_init = compute_ntk_slice(model, x0, X)
            losses = train_network(model, X, y)
            ntk_final = compute_ntk_slice(model, x0, X)
            
            # Store results
            results[str(width)]['ntk_init'].append(ntk_init.cpu().numpy())
            results[str(width)]['ntk_final'].append(ntk_final.cpu().numpy())
            results[str(width)]['losses'].append(losses)

    width_colors = {'500': 'g', '10000': 'b'}
    os.makedirs(save_dir, exist_ok=True)

    # 1. NTK Plot
    plt.figure(figsize=(6, 4))
    for width, color in width_colors.items():
        ntk_init = np.array(results[width]['ntk_init'])
        ntk_final = np.array(results[width]['ntk_final'])
        
        # Plot individual runs
        for i in range(n_inits):
            if i == 0:
                plt.plot(gamma.cpu(), ntk_init[i], color=color, alpha=1.0, linestyle='dotted',
                        label=f'n={width}, t=0', linewidth=0.5)
                plt.plot(gamma.cpu(), ntk_final[i], color=color, alpha=1.0, linestyle='solid',
                        label=f'n={width}, t=200', linewidth=0.5)
            else:
                plt.plot(gamma.cpu(), ntk_init[i], color=color, alpha=1.0, linestyle='dotted', linewidth=0.5)
                plt.plot(gamma.cpu(), ntk_final[i], color=color, alpha=1.0, linestyle='solid', linewidth=0.5)

    plt.title('NTK Values vs Angle')
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$\Theta^{(4)}(x_0, x)$')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(f'{save_dir}/ntk.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Loss Plot
    plt.figure(figsize=(6, 4))
    for width, color in width_colors.items():
        losses = np.array(results[width]['losses'])
        
        # Plot individual runs
        for i in range(n_inits):
            if i == 0:
                plt.plot(losses[i], color=color, alpha=1.0, label=f'n={width}', linewidth=0.5)
            else:
                plt.plot(losses[i], color=color, alpha=1.0, linewidth=0.5)

    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(f'{save_dir}/loss_figure1.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_figure2(widths=[50, 1000], n_points=4, n_inits=10, train_network=None, save_dir=None):
    """Generate all figures with multiple initializations."""
    # training
    gamma = torch.linspace(-3, 3, n_points + 2, device=device)
    X = torch.stack([torch.cos(gamma), torch.sin(gamma)], dim=1)
    x0 = torch.tensor([[1.0, 0.0]], device=device)
    y = 0.5 * torch.sin(2 * gamma).reshape(-1, 1)

    # testing
    gamma_ = torch.linspace(-3, 3, 100, device=device)
    X_ = torch.stack([torch.cos(gamma_), torch.sin(gamma_)], dim=1)

    results = {str(width): {
        'ntk_init': [], 'ntk_final': [], 
        'predictions': [], 'losses': [],
        'gamma': gamma_.cpu()
    } for width in widths}

    for width in widths:
        print(f"\nProcessing width {width} with {n_inits} initializations...")
        
        for init in range(n_inits):
            print(f"  Initialization {init + 1}/{n_inits}")
            torch.manual_seed(42 + init)
            
            model = NeuralNetwork(
                input_dim=2,
                hidden_dims=[width] * 3,
                output_dim=1
            )
            
            # Collect results
            ntk_init = compute_ntk_slice(model, x0, X)
            losses = train_network(model, X, y)
            ntk_final = compute_ntk_slice(model, x0, X)

            # run predictions on new samples            
            with torch.no_grad():
                predictions = model(X_)

            # Store results
            results[str(width)]['ntk_init'].append(ntk_init.cpu().numpy())
            results[str(width)]['ntk_final'].append(ntk_final.cpu().numpy())
            results[str(width)]['predictions'].append(predictions.cpu().numpy())
            results[str(width)]['losses'].append(losses)

    width_colors = {'50': 'g', '1000': 'r'}
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Function Plot
    plt.figure(figsize=(6, 4))
    for width, color in width_colors.items():
        predictions = np.array(results[width]['predictions'])
        gamma_ = np.array(results[width]['gamma'])
        ntk_final = torch.tensor(results[width]['ntk_final'][0])

        # Plot individual runs
        for i in range(n_inits):
            if i == 0:
                plt.plot(gamma_, predictions[i], color=color, alpha=1.0, 
                        label=f'n={width}', linewidth=0.5, linestyle='dotted')
            else:
                plt.plot(gamma_, predictions[i], color=color, alpha=1.0, linewidth=0.5, linestyle='dotted')

        # Plot percentiles for width=10000
        if width == '1000':
            # Compute empirical mean and std across runs
            mean = predictions.mean(axis=0)
            std = predictions.std(axis=0)
            
            # Compute percentiles
            p10 = mean + std * torch.distributions.Normal(0, 1).icdf(torch.tensor(0.1)).numpy()
            p50 = mean + std * torch.distributions.Normal(0, 1).icdf(torch.tensor(0.5)).numpy()
            p90 = mean + std * torch.distributions.Normal(0, 1).icdf(torch.tensor(0.9)).numpy()

            plt.plot(gamma_, p10, 'b--', linewidth=0.5)
            plt.plot(gamma_, p50, 'b-', label='n=∞, P50', linewidth=0.5)
            plt.plot(gamma_, p90, 'b--', label='n=∞, {P10, P90}', linewidth=0.5)

    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$f(\sin(\gamma), \cos(\gamma))$')
    plt.legend(loc='upper right')
    plt.grid(True)
    if save_dir:
        plt.savefig(f'{save_dir}/function.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Loss Plot
    plt.figure(figsize=(6, 4))
    for width, color in width_colors.items():
        losses = np.array(results[width]['losses'])
        
        # Plot individual runs
        for i in range(n_inits):
            if i == 0:
                plt.plot(losses[i], color=color, alpha=1.0, label=f'n={width}', linewidth=0.5)
            else:
                plt.plot(losses[i], color=color, alpha=1.0, linewidth=0.5)

    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(f'{save_dir}/loss_figure2.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_network_gd(model: nn.Module, X: torch.Tensor, y: torch.Tensor, 
                 n_steps: int = 200, print_freq: int = 50) -> List[float]:
    """Train the network using gradient descent."""
    # Scale learning rate with width
    learning_rate = 1.0 #0.1 / np.sqrt(model.width)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        losses.append(loss.item())
        if step % print_freq == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")
    
    return losses

class PolyakOptimizer:
    def __init__(self, parameters, lr=0.1, momentum=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p.data) for p in self.parameters]
    
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.data.zero_()
    
    def step(self):
        for i, (p, v) in enumerate(zip(self.parameters, self.velocities)):
            if p.grad is None:
                continue
            # Update velocity (momentum term)
            self.velocities[i] = self.momentum * v - self.lr * p.grad.data
            # Update parameters
            p.data.add_(self.velocities[i])

def train_network_phb(model: nn.Module, X: torch.Tensor, y: torch.Tensor, 
                 n_steps: int = 200, print_freq: int = 50) -> List[float]:
    """Train using Polyak's heavy ball method."""
    # Scale learning rate with width
    learning_rate = 1.0 #0.1 / np.sqrt(model.width)
    momentum = 0.9  # Common momentum value
    
    optimizer = PolyakOptimizer(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.MSELoss()
    losses = []

    for step in range(n_steps):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        losses.append(loss.item())
        
        if step % print_freq == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")
    
    return losses

if __name__ == "__main__":
    torch.manual_seed(42)
    plot_figure1(
        widths=[500, 10000], 
        n_points=100, 
        n_inits=10,
        train_network=train_network_gd, 
        save_dir="/playpen/ambati/ntk/gd_plots1"
    )
    plot_figure1(
        widths=[500, 10000], 
        n_points=100, 
        n_inits=10,
        train_network=train_network_phb, 
        save_dir="/playpen/ambati/ntk/hb_plots1"
    )

    plot_figure2(
        widths=[50, 1000], 
        n_points=4, 
        n_inits=10,
        train_network=train_network_gd, 
        save_dir="/playpen/ambati/ntk/gd_plots1"        
    )

    plot_figure2(
        widths=[50, 1000], 
        n_points=4, 
        n_inits=10,
        train_network=train_network_phb, 
        save_dir="/playpen/ambati/ntk/hb_plots1"
    )
