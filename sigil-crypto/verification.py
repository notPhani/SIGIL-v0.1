import torch
import numpy as np
from dataclasses import dataclass
import hashlib
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class LatticeParams:
    """Parameters defining a q-ary lattice."""
    q: int   # Prime modulus
    n: int   # Number of constraints
    m: int   # Lattice dimension (m > n)
    
    def __post_init__(self):
        if self.m <= self.n:
            raise ValueError("Require m > n")
        if self.q < 2:
            raise ValueError("q must be >= 2")
@dataclass
class QaryLattice:
    """A q-ary lattice Λ_q^⊥(A)."""
    params: LatticeParams
    A: torch.Tensor     # constraint matrix (n × m)
    basis: torch.Tensor # public basis (m × m)
    
    @property
    def device(self):
        return self.basis.device 
    
    @property
    def dimension(self):
        return self.params.m

@dataclass
class Signature:
    """Lattice-based signature."""
    s: torch.Tensor  # Signature vector (m-dimensional)
    message: str
    
    @property
    def norm(self) -> float:
        """L2 norm of signature"""
        return torch.norm(self.s.float()).item()
#----------------------------Lattice Generation----------------------------#
def _generate_constraint_matrix(
    params: LatticeParams,
    seed: str,
    device: str
) -> torch.Tensor:
    """Generate constraint matrix A = [A' | I_n] ∈ Z_q^{n×m}."""
    torch.manual_seed(hash(seed) % (2**32))
    
    A_prime = torch.randint(
        0, params.q,
        (params.n, params.m - params.n),
        dtype=torch.long,
        device=device
    )
    
    I = torch.eye(params.n, dtype=torch.long, device=device)
    A = torch.cat([A_prime, I], dim=1) % params.q
    
    return A
def _construct_public_basis(
    A: torch.Tensor,
    params: LatticeParams
) -> torch.Tensor:
    """Construct canonical public basis of Λ_q^⊥(A)."""
    n, m = A.shape
    device = A.device
    
    A_prime = A[:, :m - n]
    B = torch.zeros((m, m), dtype=torch.long, device=device)
    
    # qZ^n component
    B[:n, :n] = params.q * torch.eye(n, dtype=torch.long, device=device)
    
    # Kernel component
    B[:n, n:] = (-A_prime) % params.q
    B[n:, n:] = torch.eye(m - n, dtype=torch.long, device=device)
    
    return B
def generate_qary_lattice(
    seed: str,
    params: LatticeParams,
    device: str = "cuda"
) -> QaryLattice:
    """Generate a cryptographically hard q-ary lattice."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    A = _generate_constraint_matrix(params, seed, device)
    B = _construct_public_basis(A, params)
    
    return QaryLattice(params=params, A=A, basis=B)
#----------------------------Signature Generation----------------------------#
def _hash_message_to_syndrome(
    message: str,
    n: int,
    q: int,
    device: str
) -> torch.Tensor:
    """Hash message to target syndrome vector (n-dimensional)"""
    digest = hashlib.sha256(message.encode()).digest()
    
    seed = int.from_bytes(digest[:4], 'big')
    torch.manual_seed(seed)
    
    return torch.randint(
        0, q,
        (n,),
        dtype=torch.long,
        device=device
    )

def sign_message(
    lattice: QaryLattice,
    message: str,
    sigma: float = 2.0
) -> Signature:
    """
    Generate lattice-based signature using q-ary lattice structure.
    
    This uses the "Complex Projection / Dual Spaces" approach:
    - A = [A' | I_n]
    - Sample y ~ D_σ^{m-n} for first m-n components
    - Compute z ≡ h - A'·y (mod q) for last n components
    - Returns s = [y; z] satisfying A·s ≡ h (mod q)
    """
    device = lattice.device
    n, m = lattice.params.n, lattice.params.m
    q = lattice.params.q
    A = lattice.A
    A_prime = A[:, :m - n]
    
    # Get target syndrome (n-dimensional)
    h = _hash_message_to_syndrome(message, n, q, device)
    
    # Sample first m-n components from discrete Gaussian
    y = torch.randn(m - n, device=device) * sigma
    y = torch.round(y).to(torch.long)
    y = torch.where(y > q // 2, y - q, y)
    
    # Compute last n components to satisfy A·s ≡ h (mod q)
    z = (h - (A_prime @ y) % q) % q
    z = torch.where(z > q // 2, z - q, z)
    
    # Combine
    s = torch.cat([y, z])
    
    return Signature(s=s, message=message)

def verify_signature(
    lattice: QaryLattice,
    signature: Signature,
    noise_bound: int = 2
) -> bool:
    """Verify lattice signature: check A·s ≡ h (mod q) and norm bound."""
    A = lattice.A
    q = lattice.params.q
    n = lattice.params.n
    m = lattice.dimension
    device = lattice.device
    
    # Compute target syndrome
    h = _hash_message_to_syndrome(signature.message, n, q, device)
    
    # Check constraint: A·s ≡ h mod q
    As = (A @ signature.s) % q
    residual = (As - h) % q
    residual = torch.minimum(residual, q - residual)
    max_error = residual.abs().max().item()
    
    if max_error > noise_bound:
        return False
    
    # Check norm - realistic bounds for q-ary lattice
    norm = signature.norm
    lo = 1.0 * np.sqrt(m)
    hi = max(10.0 * np.sqrt(m), 4.0 * q * np.sqrt(n) / 4.0)
    
    return lo <= norm <= hi
#----------------------------Visualization----------------------------#
def visualize_lattice_2d(
    lattice: QaryLattice,
    num_points: int = 500,
    coeff_bound: int = 3,
    dims: tuple = (0, 1)
):
    """Generate 2D projection of lattice points."""
    B = lattice.basis
    device = B.device
    m = B.shape[1]
    
    coeffs = torch.randint(
        -coeff_bound, coeff_bound + 1,
        (num_points, m),
        device=device,
        dtype=torch.float32
    )
    
    B_float = B.float()
    points = coeffs @ B_float.T
    
    i, j = dims
    points_2d = points[:, [i, j]].cpu()
    
    return points_2d
def visualize_lattice_3d(
    lattice: QaryLattice,
    num_points: int = 500,
    coeff_bound: int = 3,
    dims: tuple = (0, 1, 2)
):
    """Generate 3D projection of lattice points."""
    B = lattice.basis
    device = B.device
    m = B.shape[1]
    
    coeffs = torch.randint(
        -coeff_bound, coeff_bound + 1,
        (num_points, m),
        device=device,
        dtype=torch.float32
    )
    
    B_float = B.float()
    points = coeffs @ B_float.T
    
    i, j, k = dims
    points_3d = points[:, [i, j, k]].cpu()
    
    return points_3d 

def plot_lattice_2d(
    lattice: QaryLattice,
    dims: tuple = (0, 1),
    save_path: str = None
):
    """Visualize 2D lattice structure only."""
    lattice_points = visualize_lattice_2d(lattice, num_points=500, coeff_bound=3, dims=dims)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Lattice points
    ax.scatter(lattice_points[:, 0], lattice_points[:, 1],
              c='steelblue', s=40, alpha=0.6, edgecolors='navy', linewidth=0.5)
    
    ax.set_xlabel(f'Dimension {dims[0]}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Dimension {dims[1]}', fontsize=12, fontweight='bold')
    ax.set_title('🛡️ Lattice Structure (2D Projection)', 
                fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()
def plot_lattice_3d(
    lattice: QaryLattice,
    dims: tuple = (0, 1, 2),
    save_path: str = None
):
    """Visualize 3D lattice structure only."""
    lattice_points = visualize_lattice_3d(lattice, num_points=800, coeff_bound=3, dims=dims)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Lattice points
    ax.scatter(lattice_points[:, 0], lattice_points[:, 1], lattice_points[:, 2],
              c='steelblue', s=30, alpha=0.5, edgecolors='navy', linewidth=0.3)
    
    ax.set_xlabel(f'Dimension {dims[0]}', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Dimension {dims[1]}', fontsize=11, fontweight='bold')
    ax.set_zlabel(f'Dimension {dims[2]}', fontsize=11, fontweight='bold')
    ax.set_title('🛡️ Lattice Structure (3D Projection)', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Make grid lines subtle
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()
def sigil():
    params = LatticeParams(q=97, n=4, m=8)
    
    lattice = generate_qary_lattice("SIGIL_demo_key", params, device="cpu")
    
    # Sign message
    message = "SIGIL: Quantum-Safe Signatures"
    
    signature = sign_message(lattice, message, sigma=2.0)
    
    valid = verify_signature(lattice, signature)
    
    if valid:
        print("✅ Signature VALID\n")
    else:
        print("❌ Signature INVALID\n")
    plot_lattice_2d(lattice, dims=(0, 1), save_path='lattice_2d.png')
    plot_lattice_3d(lattice, dims=(0, 1, 2), save_path='lattice_3d.png')
    
if __name__ == "__main__":
    sigil()