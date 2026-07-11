# SIGIL 🛡️
### **Drop-In Post-Quantum Security Layer**

> *Zero-migration quantum resistance for existing authentication systems*

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Integration](https://img.shields.io/badge/integration-15_minutes-brightgreen.svg)
![Status](https://img.shields.io/badge/status-experimental%2Fdemo-orange.svg)

**The Problem:** Quantum computers may eventually break RSA and ECDSA via Shor's algorithm; estimates for when a cryptographically relevant quantum computer will exist vary widely (commonly cited as 10-20+ years out, with significant uncertainty). Migrating your entire authentication infrastructure is expensive and risky.

**The Solution:** SIGIL is a **modular post-quantum signature layer** that sits alongside your existing crypto, providing quantum resistance without touching legacy code.

> ⚠️ **Status note:** The code in this repository is a teaching/reference implementation of a plain lattice SIS signature scheme. It is **not** a hardened, audited, production-grade post-quantum implementation. For production use, use a NIST-standardized implementation such as [liboqs](https://github.com/open-quantum-safe/liboqs) (ML-DSA / CRYSTALS-Dilithium). See the "Demo vs. Production" section below before relying on any numbers in this README.

---

## 🎯 **Why SIGIL?**

### **The Transition Challenge**

Organizations face a dilemma when preparing for quantum threats:

| Approach | Cost | Risk | Timeline |
|----------|------|------|----------|
| **Full Migration** | $$$$ | High (breaking changes) | 2-3 years |
| **Hybrid Layer** | $ | Low (additive) | 2-3 weeks |
| **Do Nothing** | $0 | Growing over time as quantum computing matures | Until broken |

**SIGIL is the hybrid approach:** Add post-quantum signatures *on top of* your existing authentication without replacing anything.

### **Core Design Principles**

1. **Non-Breaking:** Works alongside RSA/ECDSA, doesn't replace them
2. **Gradual Migration:** Opt-in per transaction, not all-or-nothing
3. **Framework Agnostic:** REST API works with any language/platform
4. **Reference Implementation:** Demonstrates lattice-based (SIS) signature concepts underlying NIST-standardized schemes like CRYSTALS-Dilithium
5. **Verifiable:** Transparent math, open-source implementation

---

## 🔄 **Classical → Post-Quantum Transition**

### **The Modular Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│  YOUR EXISTING SYSTEM (Unchanged)                           │
│  ┌────────────────────────────────────────┐                 │
│  │  User Authentication                    │                 │
│  │  -  RSA-2048 signatures                  │                 │
│  │  -  ECDSA wallet signatures               │                 │
│  │  -  JWT tokens                           │                 │
│  │  -  OAuth2 flows                         │                 │
│  └────────────────────────────────────────┘                 │
│                       │                                      │
│                       ▼                                      │
│  ┌────────────────────────────────────────┐                 │
│  │  ✨ SIGIL LAYER (Added)                │                 │
│  │  -  Generate lattice signature           │                 │
│  │  -  Verify against SIGIL policy          │                 │
│  │  -  Store parallel proof                 │                 │
│  │  -  Return verdict: ACCEPT/REJECT        │                 │
│  └────────────────────────────────────────┘                 │
│                       │                                      │
│                       ▼                                      │
│  ┌────────────────────────────────────────┐                 │
│  │  Transaction Proceeds                   │                 │
│  │  -  Original signature still validated   │                 │
│  │  -  SIGIL proof stored separately        │                 │
│  │  -  Zero breaking changes                │                 │
│  └────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### **Integration Modes**

#### **Mode 1: Advisory (Zero Risk)**

SIGIL runs in parallel but doesn't block transactions. Perfect for testing and gradual rollout.

```python
# Your existing code (unchanged)
if verify_rsa_signature(data, signature):
    # Transaction approved by classical crypto
    
    # Add SIGIL verification (non-blocking)
    sigil_verdict = sigil_api.verify(data)
    log_to_monitoring(sigil_verdict)  # Track quantum readiness
    
    process_transaction()  # Proceeds regardless
```

**Benefits:**
- Zero risk to production
- Gather metrics on quantum readiness
- Identify weak signatures before migration

#### **Mode 2: Dual Verification (Hybrid Security)**

Both classical and post-quantum signatures must pass. Provides defense-in-depth.

```python
# Existing verification
classical_valid = verify_rsa_signature(data, signature)

# Add SIGIL requirement
sigil_verdict = sigil_api.verify(data)

if classical_valid AND sigil_verdict == "ACCEPT":
    process_transaction()
else:
    reject_with_reason(classical_valid, sigil_verdict)
```

**Security guarantee:**
- An attacker generally needs to break *both* the classical scheme *and* the lattice scheme to forge a transaction
- If one system has an unknown vulnerability, the other still provides some protection
- Smooth transition path to pure post-quantum

#### **Mode 3: Post-Quantum Primary (Future-Proof)**

SIGIL becomes primary, classical signature optional.

```python
# SIGIL is primary verifier
sigil_verdict = sigil_api.verify(data)

if sigil_verdict == "ACCEPT":
    # Optional: Still check classical for backwards compat
    if legacy_clients_exist:
        verify_rsa_signature(data, signature)  # Don't block on failure
    
    process_transaction()
```

---

## 🧩 **Drop-In Integration Examples**

### **REST API (Any Language)**

```bash
# 1. Start SIGIL server (one-time)
docker run -p 8000:8000 sigil/api

# 2. Add verification to your existing flow
curl -X POST http://localhost:8000/sigil/prepare \
  -H "Content-Type: application/json" \
  -d '{
    "sender": "user@example.com",
    "receiver": "merchant@shop.com",
    "amount": "99.99",
    "message": "Order #12345"
  }'

# Response includes verdict: "ACCEPT" or "REJECT"
```

### **Python Integration**

```python
# your_app.py (existing code)
from sigil import SIGILClient  # <-- Only new import

sigil = SIGILClient("http://localhost:8000")

def process_payment(sender, receiver, amount):
    # Existing authentication
    if not authenticate_user(sender):
        return {"error": "Auth failed"}
    
    # Add SIGIL verification (2 lines)
    verdict = sigil.verify(sender, receiver, amount, "Payment")
    if verdict != "ACCEPT":
        return {"error": "Quantum signature rejected"}
    
    # Rest of your code unchanged
    charge_account(sender, amount)
    credit_account(receiver, amount)
    return {"success": True}
```

### **JavaScript/Node.js Integration**

```javascript
// server.js (existing Express app)
const sigil = require('sigil-client');

app.post('/api/transfer', async (req, res) => {
    const { from, to, amount } = req.body;
    
    // Existing JWT validation
    if (!validateJWT(req.headers.authorization)) {
        return res.status(401).json({ error: 'Unauthorized' });
    }
    
    // Add SIGIL check (3 lines)
    const verdict = await sigil.verify({
        sender: from,
        receiver: to,
        amount: amount,
        message: 'Bank transfer'
    });
    
    if (verdict !== 'ACCEPT') {
        return res.status(403).json({ error: 'Quantum signature failed' });
    }
    
    // Existing transfer logic unchanged
    await database.transfer(from, to, amount);
    res.json({ success: true });
});
```

### **Blockchain Smart Contract Integration**

```solidity
// YourContract.sol
contract SecureTransfer {
    address sigilVerifier = 0x123...;  // SIGIL oracle address
    
    function transfer(address to, uint amount, bytes memory sigilProof) public {
        // Existing checks
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Add SIGIL verification via oracle
        require(
            ISIGILOracle(sigilVerifier).verify(msg.sender, to, amount, sigilProof),
            "Post-quantum signature invalid"
        );
        
        // Transfer proceeds only if both checks pass
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
}
```

---

## 📊 **Mathematical Foundation**

### **Why Lattice Cryptography?**

| Crypto System | Quantum Vulnerable? | Security Basis | NIST Status |
|---------------|---------------------|----------------|-------------|
| RSA | ✅ YES (Shor's algorithm) | Integer factorization | Not selected as a PQC standard |
| ECDSA | ✅ YES (Shor's algorithm) | Discrete log problem | Not selected as a PQC standard |
| CRYSTALS-Dilithium (ML-DSA) | ❌ NO (no known efficient quantum attack) | Module-LWE / Module-SIS | **Selected 2022, standardized as FIPS 204 (2024)** |
| SIGIL (this repo) | ❌ NO (no known efficient quantum attack) | Plain lattice SIS problem | Educational reference; **not** a NIST-standardized scheme |

### **The SIS Problem (Simplified)**

**Challenge:** Given a random matrix \(A\) and a target vector \(h\), find a short vector \(s\) such that:

$\[
A \cdot s = h \pmod{q}
\]$

**Why it's hard:**
- Classical computers: best known attacks run in $\(2^{O(n)}\)$ time (exponential in the lattice dimension)
- Quantum computers: Grover-type search gives at most a quadratic speedup on unstructured search, so the best known attacks remain exponential (no analogue of Shor's algorithm is known for this problem)
- Best known attack in practice (lattice reduction, e.g. BKZ): roughly $\(2^{c \cdot n}\)$ operations, where \(c\) depends on the attack model (commonly cited estimates use \(c \approx 0.29\)-\(0.3\) as a rough rule of thumb, not an exact constant)

**SIGIL Parameters — demo vs. production (see note above):**
- **Demo (this repo, default):** $\(n=4, q=97\)$ — a handful of bits of security; **for learning the math only, not secure for any real use**
- **What real production-grade parameters look like:** NIST's ML-DSA (Dilithium) achieves its ~128–256-bit security levels using structured **module lattices** (Module-LWE/Module-SIS) with carefully chosen dimensions, moduli, and rejection sampling — not by simply plugging a large \(n\) into the plain SIS scheme shown in this repo. Naively scaling this repo's toy scheme to "n=256" does **not**, by itself, give 128-bit security — see the worked example below.

### **Signature Verification**

```python
# Classical RSA verification
def verify_rsa(message, signature, public_key):
    return pow(signature, e, N) == hash(message) % N  # Breakable by a sufficiently
                                                        # large quantum computer via Shor's algorithm

# SIGIL lattice verification (believed quantum-resistant at adequate parameters)
def verify_sigil(message, signature, lattice):
    h = hash(message) % q
    return (A @ signature.s) % q == h and norm(signature.s) < bound
```

---

## 🔐 **Hybrid Security Model**

### **Defense in Depth**

```
Transaction Security = Classical AND Post-Quantum (dual verification mode)

┌─────────────────────────────────────────────┐
│  Illustrative Security Timeline             │
│  (dates below are commonly cited estimates, │
│   not certainties)                          │
│                                             │
│  ████████████████████ RSA-2048 ───────────┐ │
│  ██████████████████████████████ ECDSA ───┐│ │
│  █████████████████████████████████████...  │ │
│  └─────────────────────────────────────┘   │
│         ↑                      ↑            │
│      Today              Possible quantum    │
│      (2026)             threat (est. 2030s+)│
│                                             │
│  ████████████████████████████████████████  │
│  ████ Properly-parameterized PQC ████████   │
│  █████████████████████████████████████...  │
│                                             │
│  Hybrid Mode:                               │
│  Both must pass → defense-in-depth during   │
│  the transition period                      │
└─────────────────────────────────────────────┘
```

### **Gradual Migration Path**

**Phase 1:** Advisory Mode (Months 1-3)
- SIGIL runs in shadow mode
- Collect metrics, identify issues
- Zero production risk

**Phase 2:** Dual Verification (Months 4-12)
- Both RSA and SIGIL required
- Maximum security during transition
- Build confidence in post-quantum

**Phase 3:** Post-Quantum Primary (Year 2+)
- A standards-track implementation (e.g. ML-DSA via liboqs) becomes primary verifier
- Classical signatures optional
- Full quantum resistance, contingent on using vetted, correctly-parameterized schemes

---

## 🚀 **Quick Start**

### **1. Start SIGIL Server**

```bash
# Option A: Docker (easiest)
docker run -p 8000:8000 sigil/api

# Option B: Python
pip install sigil-server
python -m sigil.server

# Server runs on http://localhost:8000
```

### **2. Add to Your App**

```python
# Install client library
pip install sigil-client

# Add 3 lines to existing code
from sigil import verify

# Before critical operation
if verify(sender, receiver, data) != "ACCEPT":
    raise SecurityError("Post-quantum verification failed")

# Rest of your code unchanged
```

### **3. Monitor Dashboard**

```bash
# View quantum readiness
curl http://localhost:8000/sigil/stats

{
  "total_verifications": 1247,
  "acceptance_rate": 0.998,
  "quantum_safe_transactions": 1245,
  "avg_verification_time_ms": 0.8
}
```

---

## 📈 **Reference Implementation Performance (demo parameters)**

The figures below are measured against this repo's **default demo parameters** (small \(n\)), not a production-security configuration. They illustrate relative overhead, not real-world post-quantum signature sizes — compare against the Dilithium column in the table further down for realistic production numbers.

| Metric | Classical RSA-2048 | SIGIL (demo params) | Notes |
|--------|---------------------|----------------------|-------|
| Signature size | 256 bytes | ~64 bytes | Demo-only; production lattice signatures (e.g. Dilithium) are **larger** than RSA, not smaller — see comparison table below |
| Signing time | 5 ms | 2.5 ms | Demo params only; not representative of production-parameter cost |
| Verification time | 0.5 ms | 0.8 ms | |
| Quantum resistant | Vulnerable to Shor's algorithm on a sufficiently capable quantum computer | Believed resistant *at production parameters*; demo params are not secure | |
| Memory usage | 2 KB | 1 KB | Demo-only |

**Takeaway:** don't extrapolate these demo numbers to production security claims. At real production parameters, lattice signatures are typically **larger** than RSA-2048 (see Dilithium's ~2420-byte signatures below), which is the normal, expected trade-off for post-quantum security today.

---

## 🔬 **Technical Deep Dive**

### **Lattice Structure**

SIGIL constructs a q-ary lattice \(\Lambda_q^{\perp}(A)\) where:

$\[
\Lambda_q^{\perp}(A) = \{ \mathbf{s} \in \mathbb{Z}^m : A \cdot \mathbf{s} \equiv 0 \pmod{q} \}
\]$

**Public Parameters (this repo's demo scheme):**
- $\(A\): Random \(n \times m\) matrix over \(\mathbb{Z}_q\)$
- $\(q\): Prime modulus (demo default: 97)$
- $\(n, m\): Lattice dimensions (demo default: \(n=4\))$

**Signature Generation:**

1. Hash message: $\(h = \text{SHA256}(m) \mod q\)$
2. Sample short vector: $\(s \sim D_{\sigma}^m\) (Gaussian distribution)$
3. Adjust to satisfy: $\(A \cdot s \equiv h \pmod{q}\)$
4. Return: $\(\sigma = (s, h)\)$

**Verification:**

$\[
\text{Accept} \iff (A \cdot s \equiv h \pmod{q}) \land (\|s\| < \beta\sqrt{m})
\]$

### **Worked Security Estimate (corrected)**

Using the rough BKZ heuristic $\(2^{0.292n}\)$ operations as an *order-of-magnitude estimate* (real security proofs are more involved and depend on the reduction quality, hardness assumptions, and concrete attack models):

For $\(n = 256\)$:

$\[
2^{0.292 \times 256} = 2^{74.8} \approx 4 \times 10^{22} \text{ operations}
\]$

**This is roughly a 75-bit security level, not 128-bit.** This matters: 75-bit security is well below the ≥128-bit floor considered adequate against realistic adversaries, and at plausible operation rates this level could be within reach of a large, well-funded classical compute cluster over a period of years — it is not "practically unbreakable." This is precisely why real standards like ML-DSA/Dilithium don't just pick \(n=256\) in a plain SIS scheme; they use structured module lattices and larger effective parameters to reach genuine 128+/192+/256-bit security levels.

**Corrected takeaway:** don't treat "n=256" as a stand-in for "128-bit security" — the two are not the same thing outside of a specific, fully-specified scheme. Any production deployment should use validated parameter sets from a standard (ML-DSA) and a vetted library (liboqs), not parameters picked by analogy to this demo.

### **Quantum Attack Resistance (accurate framing)**

**Shor's Algorithm (breaks RSA/ECDSA on a sufficiently large fault-tolerant quantum computer):**
- Input: Modulus $\(N\)$ (RSA) or curve parameters (ECDSA)
- Output: Private key material
- Offers an **exponential** quantum speedup over the best known classical factoring/discrete-log algorithms

**Lattice reduction (best known attack class on SIS/LWE-based schemes):**
- Input: Lattice basis $\(B\)$
- Output: Short vector $\(s\)$
- Best known classical and quantum algorithms are both **exponential** in the lattice dimension; Grover search gives at most a quadratic speedup on the generic search component, which does not change the exponential nature of the problem

This absence of a known exponential quantum speedup — not an absolute, parameter-independent guarantee — is the actual basis for lattice cryptography's post-quantum candidacy. Concrete security still depends entirely on choosing adequate, vetted parameters (see estimate above).

---

## 🎓 **Educational Resources**

### **Understanding Lattices (5-Minute Intro)**

A lattice is a regular grid of points in space:

```
2D Lattice Example:

    - ─────- ─────- ─────- 
    │     │     │     │
    │     │     │     │
    - ─────- ─────- ─────- 
    │     │     │     │
    │     │     │     │
    - ─────- ─────- ─────- 

Basis vectors: b₁ = (2, 0), b₂ = (0, 2)
Any point: c₁·b₁ + c₂·b₂ where c₁, c₂ ∈ ℤ
```

**Hard Problem:** Given a lattice, find the *shortest* non-zero vector.

**Why it's hard:**
- In 2D: Easy (just look at it)
- In high dimensions with well-chosen parameters: believed to be computationally infeasible for both classical and quantum computers

SIGIL's demo illustrates this idea in low dimensions purely for teaching purposes; it is not itself a secure construction.

### **SIGIL vs CRYSTALS-Dilithium (ML-DSA)**

| Feature | Dilithium / ML-DSA (NIST standard) | SIGIL (this repo, demo) |
|---------|--------------------------------------|--------------------------|
| Security | ~128/192/256-bit levels (ML-DSA-44/65/87), via structured module lattices | ~single-digit-to-tens of bits at default demo params — **not secure** |
| Signature size | ~2420 bytes (ML-DSA-44) and up | ~64 bytes at demo params (not comparable — different security level) |
| Status | FIPS 204 standardized, production-ready via vetted libraries | Teaching tool only |
| Underlying structure | Module-LWE / Module-SIS with NTT-friendly rings | Plain (unstructured) SIS |
| Use case | Deploy now, via liboqs or a vetted language binding | Learn the underlying math |

**Key Insight:** SIGIL simplifies Dilithium's design for educational purposes by dropping the module structure and using small parameters. That simplification is exactly what makes it easy to understand — and exactly what makes it insecure. For production, use [liboqs](https://github.com/open-quantum-safe/liboqs) with a full ML-DSA/Dilithium implementation.

---

## 🤝 **Migration Support**

### **Enterprise Features**

- **Gradual Rollout:** Feature flags for per-user or per-transaction enablement
- **Fallback Handling:** Automatic retry with classical-only if SIGIL unavailable
- **Audit Logging:** Detailed verification trails for compliance
- **Performance Monitoring:** Real-time metrics on quantum readiness

### **Support & Consulting**

Need help migrating? We offer:
- Architecture review sessions
- Custom integration development
- Performance tuning for high-throughput systems
- Training workshops for security teams

Contact: **support@sigil.io**

---

## 📚 **Further Reading**

- [NIST Post-Quantum Cryptography Project](https://csrc.nist.gov/projects/post-quantum-cryptography)
- [CRYSTALS-Dilithium Specification](https://pq-crystals.org/dilithium/)
- [FIPS 204: Module-Lattice-Based Digital Signature Standard](https://csrc.nist.gov/pubs/fips/204/final)
- [Lattice-Based Cryptography for Beginners](https://eprint.iacr.org/2015/938)
- [Quantum Threat Timeline (NIST Report)](https://doi.org/10.6028/NIST.IR.8413)

---

## 🎯 **TL;DR**

**Classical signatures are vulnerable to a future large-scale quantum computer. This repo is a hands-on way to learn how lattice-based signatures work — it is not a production security layer.**

```bash
# 1. Add SIGIL layer for learning/prototyping (demo params only)
pip install sigil-client

# 2. Verify transactions
verdict = sigil.verify(sender, receiver, amount)

# 3. For real quantum-readiness, swap in a vetted ML-DSA/Dilithium
#    implementation (e.g. liboqs) with production parameters
if verdict == "ACCEPT":
    process_transaction()  # Demo-only verdict — not a production security guarantee
```

---

**Built by developers exploring lattice-based cryptography — feedback and corrections welcome.**

---

## 📄 License

MIT License - Use freely, even in commercial products. **No warranty of security or fitness for any particular purpose is provided or implied** — see LICENSE for full terms.

---

## 📞 Contact

**Email:** hello@sigil.io  
**GitHub:** [github.com/sigil-crypto](https://github.com/sigil-crypto)  
**Docs:** [docs.sigil.io](https://docs.sigil.io)

---

*"Understand the math before you trust the guarantee."*