import torch
from dataclasses import dataclass

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
    """
    A q-ary lattice Λ_q^⊥(A).
    
    Attributes:
        params: Lattice parameters
        basis: Public integer basis B (m × m)
    """
    params: LatticeParams
    basis: torch.Tensor

    @property
    def device(self):
        return self.basis.device

    @property
    def dimension(self):
        return self.params.m

def _generate_constraint_matrix(
    params: LatticeParams,
    seed: str,
    device: str
) -> torch.Tensor:
    """Generate constraint matrix A = [A' | I_n] ∈ Z_q^{n×m}."""
    torch.manual_seed(hash(seed) % (2**32))

    A_prime = torch.randint(
        0,
        params.q,
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
    """
    Construct canonical public basis of Λ_q^⊥(A).
    
    Structure:
        B = [ qI_n   -A' ]
            [  0      I  ]
    
    Properties:
        - det(B) = q^n
        - Spans kernel of A mod q
        - GS collapse guaranteed
    """
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
    """
    Generate a cryptographically hard q-ary lattice.
    
    Args:
        seed: Deterministic seed
        params: Lattice parameters (q, n, m)
        device: 'cuda' or 'cpu'
    
    Returns:
        QaryLattice with basis on specified device
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("⚠ CUDA not available, using CPU")

    A = _generate_constraint_matrix(params, seed, device)
    B = _construct_public_basis(A, params)

    return QaryLattice(params=params, basis=B)
#----------------------------Lattice Generation----------------------------#
def visualize_lattice_2d(
    lattice: QaryLattice,
    num_points: int = 500,
    coeff_bound: int = 3,
    dims: tuple[int, int] = (0, 1),
):
    """
    Generate 2D projection of lattice points.
    
    Args:
        lattice: QaryLattice object
        num_points: Number of lattice points to generate
        coeff_bound: Bound on integer coefficients
        dims: Which dimensions to project to
    
    Returns:
        2D points as CPU tensor (num_points, 2)
    
    Note: Uses float arithmetic for visualization only.
    """
    B = lattice.basis
    device = B.device
    m = B.shape[1]

    coeffs = torch.randint(
        -coeff_bound,
        coeff_bound + 1,
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
    dims: tuple[int, int, int] = (0, 1, 2),
):
    """
    Generate 3D projection of lattice points.
    
    Args:
        lattice: QaryLattice object
        num_points: Number of lattice points to generate
        coeff_bound: Bound on integer coefficients
        dims: Which dimensions to project to
    
    Returns:
        3D points as CPU tensor (num_points, 3)
    
    Note: Uses float arithmetic for visualization only.
    """
    B = lattice.basis
    device = B.device
    m = B.shape[1]

    coeffs = torch.randint(
        -coeff_bound,
        coeff_bound + 1,
        (num_points, m),
        device=device,
        dtype=torch.float32
    )

    B_float = B.float()
    points = coeffs @ B_float.T

    i, j, k = dims
    points_3d = points[:, [i, j, k]].cpu()

    return points_3d
#----------------------------Lattice Visualization----------------------------#

def generate_signature(
    
)
#----------------------------Signature Generation----------------------------#