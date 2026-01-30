"""
Quantum Super-Resolution using Adaptive Non-Local Observable VQC (ANO-VQC)

This implementation accompanies the paper:
"Quantum Super-Resolution by Adaptive Non-Local Observables"

Authors: Hsin-Yi Lin, Huan-Hsin Tseng, Samuel Yen-Chi Chen, Shinjae Yoo

The code trains a Variational Quantum Circuit with trainable Hermitian observables
for image super-resolution tasks on the MNIST dataset.
"""

import os
import argparse
import itertools
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, Dataset, DataLoader
import pennylane as qml
import lpips
from torchvision import datasets, transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# ============================================================================
# Configuration
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='ANO-VQC for Quantum Super-Resolution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model architecture
    parser.add_argument('--input-size', type=int, default=4,
                        help='Input image size (creates input-size x input-size LR images)')
    parser.add_argument('--output-size', type=int, default=12,
                        help='Output image size (creates output-size x output-size HR images)')
    parser.add_argument('--n-local', type=int, default=2,
                        help='Locality of non-local observables (2 or 3)')
    parser.add_argument('--vqc-depth', type=int, default=4,
                        help='Depth of the variational quantum circuit')
    parser.add_argument('--gap', type=int, default=1,
                        help='Gap between qubits for wire combinations')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate for linear layer')
    parser.add_argument('--lr-H', type=float, default=1e-1,
                        help='Learning rate for Hermitian parameters')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loader workers')
    
    # Loss weights
    parser.add_argument('--mse-weight', type=float, default=0.3,
                        help='Weight for MSE loss component')
    parser.add_argument('--lpips-weight', type=float, default=0.7,
                        help='Weight for LPIPS loss component')
    
    # Data
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Root directory for MNIST dataset')
    parser.add_argument('--train-samples-per-class', type=int, default=1000,
                        help='Number of training samples per digit class')
    parser.add_argument('--test-samples-per-class', type=int, default=100,
                        help='Number of test samples per digit class')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory for saving results')
    parser.add_argument('--save-visualizations', action='store_true', default=True,
                        help='Save visualization plots each epoch')
    
    return parser.parse_args()


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_for_display(img_tensor):
    """Normalize image tensor from [-pi, pi] to [0, 1] for visualization."""
    img_np = img_tensor.cpu().numpy()
    min_val, max_val = img_np.min(), img_np.max()
    return (img_np - min_val) / (max_val - min_val + 1e-8)


def batch_psnr(img1, img2, max_val=1.0):
    """
    Compute PSNR for a batch of images.
    
    Args:
        img1: Ground truth tensor [N, C, H, W]
        img2: Predicted tensor [N, C, H, W]
        max_val: Maximum pixel value
        
    Returns:
        Tensor of PSNR values [N]
    """
    img1, img2 = img1.float() / max_val, img2.float() / max_val
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def batch_ssim(img1, img2, data_range=1.0):
    """
    Compute SSIM for a batch of images.
    
    Args:
        img1: Ground truth array [N, C, H, W]
        img2: Predicted array [N, C, H, W]
        data_range: Data range of images
        
    Returns:
        Array of SSIM values [N]
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    ssim_vals = []
    for i in range(img1.shape[0]):
        if img1.shape[1] == 1:  # Grayscale
            ssim_val = ssim(img1[i, 0], img2[i, 0], data_range=data_range)
        else:  # Color
            gt_img = np.transpose(img1[i], (1, 2, 0))
            pred_img = np.transpose(img2[i], (1, 2, 0))
            ssim_val = ssim(gt_img, pred_img, channel_axis=2, data_range=data_range)
        ssim_vals.append(ssim_val)
    
    return np.array(ssim_vals)


def preprocess_for_lpips(images):
    """Preprocess images for LPIPS computation."""
    # Convert from [-pi, pi] to [-1, 1]
    normalized = images / torch.pi
    
    # Convert grayscale to RGB
    if images.shape[1] == 1:
        normalized = normalized.repeat(1, 3, 1, 1)
    
    # Ensure minimum size for LPIPS
    if normalized.shape[-1] < 64:
        normalized = F.interpolate(normalized, size=(64, 64), 
                                   mode='bilinear', align_corners=False)
    return normalized


def combined_loss(predict, target, lpips_fn, mse_weight=0.3, lpips_weight=0.7):
    """
    Compute combined MSE + LPIPS loss.
    
    Args:
        predict: Predicted images
        target: Target images
        lpips_fn: LPIPS loss function
        mse_weight: Weight for MSE component
        lpips_weight: Weight for LPIPS component
        
    Returns:
        Tuple of (total_loss, mse_loss, lpips_loss)
    """
    mse_loss = F.mse_loss(predict, target)
    lpips_loss = lpips_fn(preprocess_for_lpips(predict), 
                          preprocess_for_lpips(target)).mean()
    total_loss = mse_weight * mse_loss + lpips_weight * lpips_loss
    return total_loss, mse_loss, lpips_loss


# ============================================================================
# Quantum Circuit Components
# ============================================================================

def H_layer(n_qubits):
    """Apply Hadamard gates to all qubits."""
    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)


def RY_layer(weights):
    """Apply parameterized RY rotations to all qubits."""
    for idx, w in enumerate(weights):
        qml.RY(w, wires=idx)


def entangling_layer(n_qubits):
    """Apply entangling CNOT gates."""
    for i in range(0, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


def create_hermitian(N, A, B, D):
    """
    Create a Hermitian matrix from learnable parameters.
    
    Args:
        N: Matrix dimension (2^k for k-local observable)
        A: Real part parameters for off-diagonal
        B: Imaginary part parameters for off-diagonal
        D: Diagonal parameters
        
    Returns:
        Complex Hermitian matrix
    """
    h = torch.zeros((N, N), dtype=torch.complex128, device=A.device)
    count = 0
    for i in range(1, N):
        h[i - 1, i - 1] = D[i].clone()
        for j in range(i):
            h[i, j] = A[count + j].clone() + 1j * B[count + j].clone()
        count += i
    return h + h.conj().T


# ============================================================================
# Dataset Classes
# ============================================================================

class NormalizeToPi:
    """Transform that scales tensor values from [0, 1] to [-pi, pi]."""
    def __call__(self, x):
        return x * (2 * np.pi) - np.pi


class MNISTPairDataset(Dataset):
    """Dataset returning input-output image pairs for super-resolution."""
    
    def __init__(self, subset, input_transform, label_transform):
        self.subset = subset
        self.input_transform = input_transform
        self.label_transform = label_transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image, _ = self.subset[idx]
        return self.input_transform(image), self.label_transform(image)


def create_balanced_subset(dataset, samples_per_class, num_classes=10):
    """Create a balanced subset with equal samples per class."""
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)
    indices = [idx for indices in class_indices.values() for idx in indices]
    return Subset(dataset, indices)


# ============================================================================
# ANO-VQC Model
# ============================================================================

class ANOVQC(nn.Module):
    """
    Adaptive Non-Local Observable Variational Quantum Circuit for Super-Resolution.
    
    This model uses trainable Hermitian observables acting on multiple qubits
    to expand the representational power of the quantum circuit.
    """
    
    def __init__(self, input_size, output_size, n_local, gap=1):
        super().__init__()
        
        self.n_qubits = input_size * input_size
        self.output_size = output_size
        self.n_local = n_local
        self.gap = gap
        
        # Pre-compute wire combinations for non-local observables
        self.wire_combinations = list(
            itertools.combinations(range(0, self.n_qubits, gap), n_local)
        )
        num_observables = len(self.wire_combinations)
        
        # Quantum device and circuit
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.qnode = qml.QNode(self._quantum_circuit, self.dev, interface="torch")
        
        # Hermitian observable parameters
        N = 2 ** n_local
        n_offdiag = (N * (N - 1)) // 2
        
        self.A = nn.ParameterList([
            nn.Parameter(torch.empty(n_offdiag).normal_(std=2.0)) 
            for _ in range(num_observables)
        ])
        self.B = nn.ParameterList([
            nn.Parameter(torch.empty(n_offdiag).normal_(std=2.0)) 
            for _ in range(num_observables)
        ])
        self.D = nn.ParameterList([
            nn.Parameter(torch.empty(N).normal_(std=2.0)) 
            for _ in range(num_observables)
        ])
        
        # Linear layer to map quantum measurements to HR output
        self.linear = nn.Linear(num_observables, output_size * output_size)
        
        print(f"ANO-VQC initialized:")
        print(f"  - Qubits: {self.n_qubits}")
        print(f"  - {n_local}-local observables: {num_observables}")
        print(f"  - Output: {output_size}x{output_size}")
    
    def _quantum_circuit(self, x, H_matrices):
        """Quantum circuit with adaptive non-local observables."""
        H_layer(self.n_qubits)
        RY_layer(x)
        entangling_layer(self.n_qubits)
        
        return [
            qml.expval(qml.Hermitian(H_matrices[q], wires=list(wires)))
            for q, wires in enumerate(self.wire_combinations)
        ]
    
    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # Build Hermitian matrices from parameters
        N = 2 ** self.n_local
        H_matrices = [
            create_hermitian(N, self.A[q], self.B[q], self.D[q])
            for q in range(len(self.wire_combinations))
        ]
        
        # Evaluate quantum circuit for each sample
        q_out = torch.stack([
            torch.stack(self.qnode(xi, H_matrices)).float()
            for xi in x_flat
        ])
        
        return self.linear(q_out)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer_params, optimizer_H, lpips_fn, 
                mse_weight, lpips_weight, device):
    """Train for one epoch."""
    model.train()
    metrics = {'loss': 0, 'mse': 0, 'lpips': 0, 'psnr': 0, 'ssim': 0, 'n': 0}
    
    for X, y in tqdm(loader, desc="Training"):
        X, y = X.to(device), y.to(device)
        batch_size = len(y)
        
        optimizer_params.zero_grad()
        optimizer_H.zero_grad()
        
        pred = model(X).reshape(y.shape)
        loss, mse, lpips_val = combined_loss(pred, y, lpips_fn, mse_weight, lpips_weight)
        
        loss.backward()
        optimizer_params.step()
        optimizer_H.step()
        
        # Compute metrics
        with torch.no_grad():
            psnr_val = batch_psnr(pred, y, max_val=torch.pi).mean()
            ssim_val = batch_ssim(pred.cpu().numpy(), y.cpu().numpy()).mean()
        
        metrics['loss'] += loss.item() * batch_size
        metrics['mse'] += mse.item() * batch_size
        metrics['lpips'] += lpips_val.item() * batch_size
        metrics['psnr'] += psnr_val.item() * batch_size
        metrics['ssim'] += ssim_val * batch_size
        metrics['n'] += batch_size
    
    return {k: v / metrics['n'] for k, v in metrics.items() if k != 'n'}


@torch.no_grad()
def evaluate(model, loader, lpips_fn, mse_weight, lpips_weight, device):
    """Evaluate model on a dataset."""
    model.eval()
    metrics = {'loss': 0, 'mse': 0, 'lpips': 0, 'psnr': 0, 'ssim': 0, 'n': 0}
    
    for X, y in tqdm(loader, desc="Evaluating"):
        X, y = X.to(device), y.to(device)
        batch_size = len(y)
        
        pred = model(X).reshape(y.shape)
        loss, mse, lpips_val = combined_loss(pred, y, lpips_fn, mse_weight, lpips_weight)
        
        psnr_val = batch_psnr(pred, y, max_val=torch.pi).mean()
        ssim_val = batch_ssim(pred.cpu().numpy(), y.cpu().numpy()).mean()
        
        metrics['loss'] += loss.item() * batch_size
        metrics['mse'] += mse.item() * batch_size
        metrics['lpips'] += lpips_val.item() * batch_size
        metrics['psnr'] += psnr_val.item() * batch_size
        metrics['ssim'] += ssim_val * batch_size
        metrics['n'] += batch_size
    
    return {k: v / metrics['n'] for k, v in metrics.items() if k != 'n'}


def save_visualization(model, test_subset, input_transform, label_transform, 
                       input_size, output_size, epoch, metrics, output_dir, device):
    """Save visualization of SR results for each digit."""
    import matplotlib.pyplot as plt
    
    # Get one example per digit
    examples = {}
    for idx in range(len(test_subset)):
        image, label = test_subset[idx]
        if label not in examples:
            examples[label] = image
        if len(examples) == 10:
            break
    
    fig, axes = plt.subplots(10, 3, figsize=(9, 30))
    
    model.eval()
    for i, digit in enumerate(sorted(examples.keys())):
        image = examples[digit]
        
        input_tensor = input_transform(image).unsqueeze(0).to(device)
        gt_tensor = label_transform(image)
        
        with torch.no_grad():
            pred = model(input_tensor)
        pred_tensor = pred.view(output_size, output_size)
        
        # Convert for display
        input_np = normalize_for_display(input_tensor.squeeze().view(input_size, input_size))
        gt_np = normalize_for_display(gt_tensor.view(output_size, output_size))
        pred_np = normalize_for_display(pred_tensor)
        
        axes[i, 0].imshow(input_np, cmap='gray', interpolation='nearest')
        axes[i, 0].set_title(f"Input {input_size}×{input_size}" if i == 0 else "")
        axes[i, 0].set_ylabel(f"Digit {digit}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(gt_np, cmap='gray', interpolation='nearest')
        axes[i, 1].set_title(f"Ground Truth {output_size}×{output_size}" if i == 0 else "")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(pred_np, cmap='gray', interpolation='nearest')
        axes[i, 2].set_title(f"Predicted {output_size}×{output_size}" if i == 0 else "")
        axes[i, 2].axis("off")
    
    fig.suptitle(f"Epoch {epoch+1} | PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.3f}", 
                 fontsize=14, y=1.01)
    plt.tight_layout()
    
    filename = f"visualization_epoch{epoch+1:02d}_psnr{metrics['psnr']:.2f}_ssim{metrics['ssim']:.3f}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print(f"Configuration: {args.input_size}×{args.input_size} → {args.output_size}×{args.output_size}")
    print(f"Non-local observables: {args.n_local}-local")
    
    # Data transforms
    input_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        NormalizeToPi(),
    ])
    
    label_transform = transforms.Compose([
        transforms.Resize((args.output_size, args.output_size)),
        transforms.ToTensor(),
        NormalizeToPi(),
    ])
    
    # Load datasets
    print("\nLoading MNIST dataset...")
    train_dataset = datasets.MNIST(root=args.data_root, train=True, download=True)
    test_dataset = datasets.MNIST(root=args.data_root, train=False, download=True)
    
    # Create balanced subsets
    train_subset = create_balanced_subset(train_dataset, args.train_samples_per_class)
    test_subset = create_balanced_subset(test_dataset, args.test_samples_per_class)
    
    train_pair_dataset = MNISTPairDataset(train_subset, input_transform, label_transform)
    test_pair_dataset = MNISTPairDataset(test_subset, input_transform, label_transform)
    
    train_loader = DataLoader(train_pair_dataset, batch_size=args.batch_size, 
                              shuffle=True, pin_memory=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_pair_dataset, batch_size=args.batch_size, 
                             shuffle=False, pin_memory=True, num_workers=args.num_workers)
    
    print(f"Training samples: {len(train_pair_dataset)}")
    print(f"Test samples: {len(test_pair_dataset)}")
    
    # Initialize model
    print("\nInitializing ANO-VQC model...")
    model = ANOVQC(
        input_size=args.input_size,
        output_size=args.output_size,
        n_local=args.n_local,
        gap=args.gap
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Setup optimizers (separate learning rates for Hermitian params and linear layer)
    H_params = [p for n, p in model.named_parameters() if any(x in n for x in ['A', 'B', 'D'])]
    linear_params = [p for n, p in model.named_parameters() if 'linear' in n]
    
    optimizer_H = torch.optim.Adam(H_params, lr=args.lr_H)
    optimizer_params = torch.optim.Adam(linear_params, lr=args.lr)
    
    # LPIPS loss function
    lpips_fn = lpips.LPIPS(net='vgg').to(DEVICE)
    lpips_fn.eval()
    for param in lpips_fn.parameters():
        param.requires_grad = False
    
    # Training history
    history = {'train': [], 'test': []}
    best_test_loss = float('inf')
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print('='*60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer_params, optimizer_H, lpips_fn,
            args.mse_weight, args.lpips_weight, DEVICE
        )
        history['train'].append(train_metrics)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"MSE: {train_metrics['mse']:.4f}, LPIPS: {train_metrics['lpips']:.4f}")
        print(f"        PSNR: {train_metrics['psnr']:.2f}, SSIM: {train_metrics['ssim']:.4f}")
        
        # Evaluate
        test_metrics = evaluate(
            model, test_loader, lpips_fn, args.mse_weight, args.lpips_weight, DEVICE
        )
        history['test'].append(test_metrics)
        
        print(f"Test  - Loss: {test_metrics['loss']:.4f}, "
              f"MSE: {test_metrics['mse']:.4f}, LPIPS: {test_metrics['lpips']:.4f}")
        print(f"        PSNR: {test_metrics['psnr']:.2f}, SSIM: {test_metrics['ssim']:.4f}")
        
        # Save best model
        if test_metrics['loss'] < best_test_loss:
            best_test_loss = test_metrics['loss']
            model_path = os.path.join(
                args.output_dir,
                f"best_model_{args.n_local}local_{args.input_size}to{args.output_size}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_H_state_dict': optimizer_H.state_dict(),
                'optimizer_params_state_dict': optimizer_params.state_dict(),
                'metrics': test_metrics,
                'args': vars(args)
            }, model_path)
            print(f"✓ Saved best model (loss: {best_test_loss:.4f})")
        
        # Save visualization
        if args.save_visualizations:
            save_visualization(
                model, test_subset, input_transform, label_transform,
                args.input_size, args.output_size, epoch, test_metrics, 
                args.output_dir, DEVICE
            )
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.npz')
    np.savez(
        history_path,
        train_loss=[m['loss'] for m in history['train']],
        train_mse=[m['mse'] for m in history['train']],
        train_lpips=[m['lpips'] for m in history['train']],
        train_psnr=[m['psnr'] for m in history['train']],
        train_ssim=[m['ssim'] for m in history['train']],
        test_loss=[m['loss'] for m in history['test']],
        test_mse=[m['mse'] for m in history['test']],
        test_lpips=[m['lpips'] for m in history['test']],
        test_psnr=[m['psnr'] for m in history['test']],
        test_ssim=[m['ssim'] for m in history['test']],
    )
    print(f"\nTraining complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()