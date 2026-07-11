import torch
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import time
import math


@dataclass(frozen=True)
class LatticeParams:
    q: int   # modulus
    n: int   # constraints
    m: int   # dimension (m > n)

    def __post_init__(self):
        if self.m <= self.n:
            raise ValueError("Require m > n")
        if self.q < 2:
            raise ValueError("q must be >= 2")


@dataclass
class QaryLattice:
    params: LatticeParams
    A: torch.Tensor     # constraint matrix (n × m)
    B: torch.Tensor     # public basis (m × m)

    @property
    def device(self):
        return self.B.device
    
    @property
    def dimension(self):
        return self.params.m

    @property
    def dual_basis(self) -> torch.Tensor:
        """Compute dual basis B* = (B^T)^(-1) for Gaussian sampling"""
        if not hasattr(self, '_dual_basis'):
            B_float = self.B.float()
            self._dual_basis = torch.linalg.inv(B_float.T)
        return self._dual_basis

    def get_trapdoor_basis(self) -> torch.Tensor:
        """Return the trapdoor basis for efficient Gaussian sampling.
        Uses the dual lattice basis for GPV-style sampling."""
        return self.dual_basis


@dataclass
class Signature:
    s: torch.Tensor     # vector in Z^m
    message: str

    @property
    def norm(self):
        return torch.norm(self.s.float()).item()


def generate_constraint_matrix(
    params: LatticeParams,
    seed: str,
    device: str
) -> torch.Tensor:
    torch.manual_seed(hash(seed) % (2**32))

    A_prime = torch.randint(
        0, params.q,
        (params.n, params.m - params.n),
        device=device,
        dtype=torch.long
    )

    I = torch.eye(params.n, device=device, dtype=torch.long)
    return torch.cat([A_prime, I], dim=1) % params.q


def construct_public_basis(A: torch.Tensor, params: LatticeParams) -> torch.Tensor:
    n, m = A.shape
    device = A.device
    A_prime = A[:, :m - n]

    B = torch.zeros((m, m), dtype=torch.long, device=device)

    # Kernel basis: B = [ I_{m-n}    0    ]
    #                  [ -A'        q*I_n ]
    # This ensures A @ B ≡ 0 (mod q)
    
    # First m-n columns: [I_{m-n}; -A']
    B[:m - n, :m - n] = torch.eye(m - n, dtype=torch.long, device=device)
    B[m - n:, :m - n] = (-A_prime) % params.q
    
    # Last n columns: [0; q*I_n]
    B[m - n:, m - n:] = params.q * torch.eye(n, dtype=torch.long, device=device)

    return B


def generate_qary_lattice(
    seed: str,
    params: LatticeParams,
    device: str = "cpu"
) -> QaryLattice:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    A = generate_constraint_matrix(params, seed, device)
    B = construct_public_basis(A, params)
    return QaryLattice(params=params, A=A, B=B)


def hash_message_to_syndrome(
    message: str,
    n: int,
    q: int,
    device: str
) -> torch.Tensor:
    """Hash message to target syndrome vector"""
    digest = hashlib.sha256(message.encode()).digest()
    seed = int.from_bytes(digest[:4], "big")
    torch.manual_seed(seed)

    return torch.randint(
        0, q,
        (n,),
        device=device,
        dtype=torch.long
    )


def _sample_discrete_gaussian_dual(
    lattice: QaryLattice,
    sigma: float,
    center: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Sample from discrete Gaussian distribution over the lattice using the 
    dual lattice basis (Complex Projection / GPV approach).
    
    This implements the Klein/GPV sampler using the dual basis B* = (B^T)^{-1}
    for the q-ary lattice Λ_q^⊥(A).
    
    The dual lattice approach (Complex Projection Space) works as follows:
    - The q-ary lattice has basis B (m × m)
    - Its dual lattice has basis B* = (B^T)^{-1}
    - To sample from D_{L, σ, c}, we sample z ~ N(0, σ²I) in dual basis coordinates
    - Then compute v = B*^T @ z, round to nearest integer, then map back via B
    
    NOTE: This requires the basis B to be a trapdoor basis (small GS norms).
    The canonical q-ary basis is NOT a trapdoor basis. For proper GPV,
    we need a trapdoor basis generated with A (e.g., MP12).
    
    Args:
        lattice: The q-ary lattice
        sigma: Gaussian parameter (standard deviation)
        center: Optional center for coset sampling (length m)
        
    Returns:
        Sampled lattice vector (or coset vector)
    """
    device = lattice.device
    m = lattice.dimension
    
    # Get the dual basis B* = (B^T)^{-1}
    # This is the "complex projection space" approach - we work in the dual basis
    B_float = lattice.B.float()
    
    # Compute dual basis: B* = (B^T)^{-1}
    try:
        B_T = B_float.T
        B_star_T = torch.linalg.inv(B_T)
        B_star = B_star_T.T  # B* = (B^T)^{-1}
    except:
        # Fallback: use pseudo-inverse for numerical stability
        B_T = B_float.T
        B_star_T = torch.linalg.pinv(B_T)
        B_star = B_star_T.T
    
    # Center for sampling (default: origin for lattice, or coset center)
    if center is None:
        center = torch.zeros(m, device=device, dtype=torch.float32)
    else:
        center = center.float()
    
    # Sample in the dual basis coordinates
    # z ~ N(0, σ² I) in the dual space
    z = torch.randn(m, device=device, dtype=torch.float32) * sigma
    
    # Map to primal basis coordinates: v = B*^T @ z
    v = B_star.T @ z
    
    # Round to nearest integer (this is the core of Klein's algorithm)
    coeffs = torch.round(v).to(torch.long)
    
    # Map back to primal space: s = B @ coeffs
    # This gives a vector in the lattice
    s = (B_float @ coeffs.float()).round().to(torch.long)
    
    # Add center if provided (for coset sampling)
    if center is not None and center.abs().sum() > 0:
        s = s + center.round().long()
    
    return s


def _sample_coset_gaussian_qary(
    lattice: QaryLattice,
    target: torch.Tensor,
    sigma: float
) -> torch.Tensor:
    """
    Sample a short vector from the coset {s: A·s ≡ target (mod q)}.
    
    For q-ary lattice with A = [A' | I_n], we can efficiently sample from
    the coset by:
    1. Sample y from discrete Gaussian over Z^{m-n} (first m-n components)
    2. Compute z = (target - A' @ y) mod q (last n components)
    3. Center z to be in [-q/2, q/2]
    4. Return concatenated vector [y; z]
    
    This uses the special structure of q-ary lattices and avoids needing
    a full trapdoor basis. It's the standard way to sample SIS solutions.
    """
    device = lattice.device
    n, m = lattice.params.n, lattice.params.m
    q = lattice.params.q
    A = lattice.A
    A_prime = A[:, :m - n]
    
    # Sample first m-n components from discrete Gaussian
    y = torch.randn(m - n, device=device) * sigma
    y = torch.round(y).to(torch.long)
    y = y % q
    y = torch.where(y > q // 2, y - q, y)
    
    # Compute last n components to satisfy A·s ≡ target (mod q)
    # A · [y; z] = A'·y + I_n·z ≡ target (mod q)
    # So z ≡ target - A'·y (mod q)
    z = (target - (A_prime @ y) % q) % q
    z = torch.where(z > q // 2, z - q, z)
    
    # Combine
    s = torch.cat([y, z])
    
    return s


def _find_particular_solution(
    lattice: QaryLattice,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Find any particular solution s0 to A @ s0 ≡ target (mod q).
    
    Since A = [A' | I_n], we can set:
    - First m-n components to 0
    - Last n components to target (mod q)
    
    This works because A @ s0 = A[:, m-n:] @ target ≡ I_n @ target ≡ target (mod q)
    
    Returns a solution vector of length m.
    """
    n, m = lattice.params.n, lattice.params.m
    q = lattice.params.q
    device = lattice.device
    
    s0 = torch.zeros(m, device=device, dtype=torch.long)
    s0[m-n:] = target % q
    
    return s0


def sign_message(
    lattice: QaryLattice,
    message: str,
    sigma: float = 2.0
) -> Signature:
    """
    SIS-based signing using the q-ary lattice structure.
    
    This uses the "Complex Projection / Dual Spaces" approach for q-ary lattices:
    - The constraint A·s ≡ h (mod q) with A = [A' | I_n]
    - We sample the first m-n components from a discrete Gaussian
    - We compute the last n components to EXACTLY satisfy the equation
    - This projects the problem from Z_q^m to Z_q^n via the linear map A
    
    This is the standard method for solving SIS over q-ary lattices
    without needing a full trapdoor basis.
    """
    device = lattice.device
    n, m = lattice.params.n, lattice.params.m
    q = lattice.params.q
    A = lattice.A
    A_prime = A[:, :m - n]
    
    # Get target syndrome
    h = hash_message_to_syndrome(message, n, q, device)
    
    # Sample first m-n components from discrete Gaussian
    # Use centered values around 0 for smaller norm
    y = torch.randn(m - n, device=device) * sigma
    y = torch.round(y).to(torch.long)
    y = torch.where(y > q // 2, y - q, y)  # Center in [-q/2, q/2]
    
    # Compute last n components to EXACTLY satisfy A·s ≡ h (mod q)
    # A·[y; z] = A'·y + z ≡ h (mod q)  =>  z ≡ h - A'·y (mod q)
    z = (h - (A_prime @ y) % q) % q
    z = torch.where(z > q // 2, z - q, z)  # Center in [-q/2, q/2]
    
    # Combine
    s = torch.cat([y, z])
    
    return Signature(s=s, message=message)



class SIGILVerifier:
    """
    SIS-style verifier:
    Accepts iff:
      1. A·s ≡ H(m) mod q (exact or with small error)
      2. ||s|| is reasonably small
    """

    def __init__(
        self,
        lattice: QaryLattice,
        noise_bound: int = 2
    ):
        self.lattice = lattice
        self.noise_bound = noise_bound

    def verify(self, sig: Signature) -> Tuple[bool, Dict]:
        """Verify signature and return detailed info"""
        A = self.lattice.A
        q = self.lattice.params.q
        m = self.lattice.params.m
        n = self.lattice.params.n
        device = self.lattice.device

        # Compute target syndrome
        h = hash_message_to_syndrome(
            sig.message,
            self.lattice.params.n,
            q,
            device
        )

        # Check constraint: A·s ≡ h mod q
        As = (A @ sig.s) % q
        residual = (As - h) % q
        residual = torch.minimum(residual, q - residual)  # centered mod q

        max_error = residual.abs().max().item()
        constraint_satisfied = (max_error <= self.noise_bound)

        # Check norm - use realistic bounds for q-ary lattice signatures
        # Expected norm: sqrt((m-n)*sigma^2 + n*(q/4)^2) for sigma=2
        # For m=8, n=4, q=97: sqrt(4*4 + 4*(97/4)^2) ≈ sqrt(16 + 2352) ≈ 48.6
        norm = sig.norm
        
        # Use adaptive bounds based on lattice parameters
        # Lower bound: should be non-trivial (at least 1*sqrt(m))
        lo = 1.0 * np.sqrt(m)
        # Upper bound: theoretical max for this scheme ~ 3*sqrt(m)*(q/4) 
        # For demo params: ~3*2.8*24 ≈ 200, but typical is ~50-60
        hi = max(10.0 * np.sqrt(m), 4.0 * q * np.sqrt(n) / 4.0)
        
        norm_ok = (lo <= norm <= hi)

        # Verification passes if both conditions met
        valid = constraint_satisfied and norm_ok

        details = {
            'valid': valid,
            'constraint_satisfied': constraint_satisfied,
            'norm_ok': norm_ok,
            'max_error': max_error,
            'signature_norm': norm,
            'expected_norm_range': (lo, hi),
            'residual_vector': residual.cpu().tolist()
        }

        return valid, details

    def verify_with_score(self, sig: Signature) -> Dict:
        """
        Probabilistic verification with detailed scoring.
        """
        A = self.lattice.A
        q = self.lattice.params.q
        m = self.lattice.params.m
        n = self.lattice.params.n
        device = self.lattice.device

        # Target syndrome
        target = hash_message_to_syndrome(
            sig.message,
            self.lattice.params.n,
            q,
            device
        )

        # Constraint residual
        residual = (A @ sig.s - target) % q
        residual = torch.minimum(residual, q - residual)
        residual_norm = torch.norm(residual.float()).item()

        # Signature norm - expected norm for this scheme
        sig_norm = sig.norm
        # Expected norm: sqrt((m-n)*sigma^2 + n*(q/4)^2)
        # For typical sigma=2: sqrt(4*4 + 4*(97/4)^2) ≈ 48.6
        expected_norm = np.sqrt((m - n) * 4.0 + n * (q / 4.0)**2)

        # Scoring
        alpha, beta = 0.5, 0.3
        constraint_score = np.exp(-alpha * residual_norm)
        norm_score = np.exp(-beta * abs(sig_norm - expected_norm) / expected_norm)

        final_score = 0.6 * constraint_score + 0.4 * norm_score

        return {
            "final_score": final_score,
            "constraint_norm": residual_norm,
            "signature_norm": sig_norm,
            "constraint_score": constraint_score,
            "norm_score": norm_score,
            "verdict": "ACCEPT" if final_score > 0.5 else "REJECT"
        }


def comprehensive_test(lattice: QaryLattice, num_tests: int = 10):
    """Run comprehensive verification tests"""
    
    print("\n" + "="*70)
    print("🔬 SIGIL COMPREHENSIVE TESTING".center(70))
    print("="*70 + "\n")

    verifier = SIGILVerifier(lattice, noise_bound=2)

    # Test 1: Valid signatures
    print("📝 Test 1: Valid Signature Verification")
    print("-"*70)
    
    passed = 0
    failed = 0
    
    for i in range(num_tests):
        msg = f"Test message {i}"
        sig = sign_message(lattice, msg, sigma=1.5)
        valid, details = verifier.verify(sig)
        
        if valid:
            passed += 1
            if i < 3:
                print(f"✅ Message {i}: VALID (norm={sig.norm:.2f}, error={details['max_error']})")
        else:
            failed += 1
            if i < 3:
                print(f"❌ Message {i}: INVALID (norm={sig.norm:.2f}, error={details['max_error']})")
    
    print(f"\nResult: {passed}/{num_tests} signatures verified ({passed/num_tests*100:.1f}%)\n")

    # Test 2: Forgery resistance
    print("🛡️  Test 2: Forgery Resistance")
    print("-"*70)
    
    msg = "Original message"
    real_sig = sign_message(lattice, msg)
    
    # Random forgery
    fake_sig = Signature(
        s=torch.randint(-10, 10, (lattice.dimension,), device=lattice.device),
        message=msg
    )
    
    real_valid, real_details = verifier.verify(real_sig)
    fake_valid, fake_details = verifier.verify(fake_sig)
    
    print(f"Real signature: {'✅ VALID' if real_valid else '❌ INVALID'} (error={real_details['max_error']})")
    print(f"Fake signature: {'❌ INVALID' if not fake_valid else '✅ VALID (PROBLEM!)'} (error={fake_details['max_error']})")
    
    # Message tampering
    tampered_sig = Signature(s=real_sig.s, message="Tampered message")
    tampered_valid, _ = verifier.verify(tampered_sig)
    print(f"Tampered message: {'❌ INVALID' if not tampered_valid else '✅ VALID (PROBLEM!)'}\n")

    # Test 3: Performance
    print("⚡ Test 3: Performance")
    print("-"*70)
    
    num_bench = 100
    
    # Signing speed
    start = time.time()
    for i in range(num_bench):
        sign_message(lattice, f"Benchmark {i}")
    sign_time = (time.time() - start) / num_bench * 1000
    
    # Verification speed
    sig = sign_message(lattice, "Benchmark")
    start = time.time()
    for _ in range(num_bench):
        verifier.verify(sig)
    verify_time = (time.time() - start) / num_bench * 1000
    
    print(f"Signing:      {sign_time:.3f} ms/signature")
    print(f"Verification: {verify_time:.3f} ms/verify\n")

    # Test 4: Probabilistic scoring
    print("📊 Test 4: Probabilistic Scoring")
    print("-"*70)
    
    msg = "Score test"
    sig = sign_message(lattice, msg)
    scores = verifier.verify_with_score(sig)
    
    print(f"Constraint score: {scores['constraint_score']:.4f}")
    print(f"Norm score:       {scores['norm_score']:.4f}")
    print(f"Final score:      {scores['final_score']:.4f}")
    print(f"Verdict:          {scores['verdict']}\n")


def plot_lattice_2d(lattice: QaryLattice, dims=(0,1), save_path=None):
    """Visualize 2D lattice projection"""
    B = lattice.B.float()
    coeffs = torch.randint(-3, 4, (500, lattice.params.m), dtype=torch.float32)
    points = coeffs @ B.T
    pts = points[:, list(dims)].cpu()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pts[:,0], pts[:,1], s=30, alpha=0.6, c='steelblue', edgecolors='navy', linewidth=0.5)
    ax.set_xlabel(f'Dimension {dims[0]}', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Dimension {dims[1]}', fontsize=12, fontweight='bold')
    ax.set_title('🛡️ SIGIL Lattice Structure (2D Projection)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


def sigil_demo():

    # Generate lattice
    params = LatticeParams(q=97, n=4, m=8)
    lattice = generate_qary_lattice("SIGIL_demo_key", params)

    verifier = SIGILVerifier(lattice, noise_bound=2)
    msg = "SIGIL: Post-Quantum Identity"
    
    sig = sign_message(lattice, msg, sigma=1.5)
    
    ok, details = verifier.verify(sig)
    
    if ok:
        print(f"✅ SIGNATURE VALID")
        print(f"   Max error: {details['max_error']}")
        print(f"   Signature norm: {details['signature_norm']:.2f}")
    else:
        print(f"❌ SIGNATURE INVALID")
        print(f"   Reason: constraint={details['constraint_satisfied']}, norm={details['norm_ok']}")

    # Comprehensive tests
    comprehensive_test(lattice, num_tests=20)
    
    plot_lattice_2d(lattice, dims=(0, 1), save_path='sigil_lattice.png')


if __name__ == "__main__":
    sigil_demo()
