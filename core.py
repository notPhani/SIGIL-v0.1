import torch
import numpy as np
import hashlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict
import time


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

    # qZ^n
    B[:n, :n] = params.q * torch.eye(n, dtype=torch.long, device=device)

    # kernel block
    B[:n, n:] = (-A_prime) % params.q
    B[n:, n:] = torch.eye(m - n, dtype=torch.long, device=device)

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


def sign_message(
    lattice: QaryLattice,
    message: str,
    sigma: float = 1.5  # Reduced from 2.0
) -> Signature:
    """
    SIS-based signing with controlled norm
    """
    device = lattice.device
    n, m = lattice.params.n, lattice.params.m
    q = lattice.params.q
    A = lattice.A

    # Get target syndrome
    h = hash_message_to_syndrome(message, n, q, device)

    # Start with VERY small random vector
    torch.manual_seed(hash(message) % (2**32))
    s = torch.randint(-2, 3, (m,), device=device, dtype=torch.long)  # Range: -2 to 2
    
    # Add minimal Gaussian noise
    noise = torch.randn(m, device=device) * sigma
    s = s + noise.round().long()
    
    # FIXED: Use centered reduction for adjustment
    residual = (A[:, :m-n] @ s[:m-n]) % q
    adjustment = (h - residual) % q
    
    # Center the adjustment values around 0 (not 0 to q)
    adjustment = torch.where(adjustment > q // 2, adjustment - q, adjustment)
    
    s[m-n:] = adjustment

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

        # Check norm
        norm = sig.norm
        lo = 0.5 * np.sqrt(m)
        hi = 8.0 * np.sqrt(m)
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

        # Signature norm
        sig_norm = sig.norm
        expected_norm = np.sqrt(m)

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
