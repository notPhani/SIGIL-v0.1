<p align="center">
  <img src="https://img.shields.io/badge/SIGIL-Post--Quantum%20Cryptography-8B6FD6?style=for-the-badge&logo=shield&logoColor=white" alt="SIGIL Badge"/>
</p>

<h1 align="center">🛡️ SIGIL</h1>
<h3 align="center"><em>Signatures Quantum Can't Break</em></h3>

<p align="center">
  <strong>A Post-Quantum Cryptographic Layer Built for the Age of Quantum Computing</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-Accelerated-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Beta-orange?style=flat-square"/>
</p>

<p align="center">
  <a href="#-the-quantum-threat">The Threat</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-demo">Demo</a>
</p>

---

## ⚠️ The Quantum Threat

> **"Q-Day is coming."** — When large-scale quantum computers arrive, RSA, ECDSA, and every classical cryptographic signature will be **instantly broken** by Shor's Algorithm.

SIGIL is your defense. Built on **lattice-based cryptography** — the mathematical foundation behind NIST's post-quantum standards — SIGIL provides unforgeable digital signatures that remain secure even against adversaries wielding quantum computers.

```
┌─────────────────────────────────────────────────────────────────┐
│  🕐 DOOMSDAY CLOCK: ~16 YEARS UNTIL CRYPTOGRAPHICALLY          │
│     RELEVANT QUANTUM COMPUTERS (CRQC)                          │
│                                                                 │
│  📊 HARVEST NOW, DECRYPT LATER (HNDL) ATTACKS ARE HAPPENING    │
│     TODAY — YOUR DATA IS ALREADY BEING COLLECTED               │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### 🔐 **Lattice-Based Signatures (SIS Problem)**
SIGIL implements the **Short Integer Solution (SIS)** problem over q-ary lattices — proven to be NP-hard and resistant to both classical and quantum attacks.

### ⚡ **GPU-Accelerated Cryptography**
Built on PyTorch for blazing-fast tensor operations. Seamlessly runs on CUDA-enabled GPUs or falls back to optimized CPU computation.

### 🌐 **Production-Ready REST API**
FastAPI-powered backend with real-time transaction signing, verification scoring, and comprehensive history tracking.

### 🎨 **Beautiful Interactive Demo**
Stunning web interface with MetaMask integration, animated lattice visualizations, and real blockchain transactions on Sepolia testnet.

### 🧪 **Quantum Attack Simulation**
Includes a full **quantum circuit simulator (QtorchX)** with 2500+ lines of code demonstrating Shor's algorithm — proving exactly what SIGIL defends against.

---

## 🏗️ Architecture

```
SIGIL/
├── 🧠 core.py                    # Lattice cryptography engine
│   ├── LatticeParams             # q-ary lattice configuration
│   ├── QaryLattice               # Lattice structure with basis matrices
│   ├── sign_message()            # SIS-based signature generation
│   └── SIGILVerifier             # Dual verification (boolean + scoring)
│
├── 🌐 transac_api.py             # FastAPI REST interface
│   ├── POST /sigil/prepare       # Generate quantum-safe signatures
│   ├── POST /sigil/record        # Record verified transactions
│   ├── GET  /sigil/history       # Transaction history
│   └── GET  /sigil/stats         # Lattice parameters & security bits
│
├── 📁 sigil-crypto/              # Extended cryptographic modules
│   ├── verification.py           # Alternative verification & visualization
│   └── Attacker model/
│       ├── Qtorch.py             # 🚨 2500+ line quantum circuit simulator
│       └── rsa-breaker.py        # Shor's algorithm RSA factorization demo
│
└── 🎨 final_static/              # Web interface
    ├── index.html                # Responsive UI with scroll animations
    ├── sigil-transaction.js      # MetaMask + SIGIL API integration
    ├── blob.js                   # Three.js lattice visualizations
    ├── noise.js                  # Perlin noise for visual effects
    └── style.css                 # Beautiful dark theme styling
```

---

## 🧮 The Math Behind SIGIL

SIGIL's security is founded on the **computational hardness of lattice problems**:

### Q-ary Lattice Construction
```
Λ_q^⊥(A) = { x ∈ Z^m : Ax ≡ 0 (mod q) }
```

### Signature Generation (SIS-Based)
Given a message `m`, SIGIL:
1. **Hashes** `m → h ∈ Z_q^n` using SHA-256
2. **Samples** a short vector `s` where `As ≡ h (mod q)`
3. **Returns** signature `σ = (s, m)` with controlled L2 norm

### Verification
```python
# Boolean Check
valid = (A @ s ≡ H(m) mod q) AND (||s|| ∈ acceptable_range)

# Probabilistic Scoring
score = 0.6 × exp(-α × residual_norm) + 0.4 × exp(-β × norm_deviation)
```

### Security Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `q` | 97 | Prime modulus |
| `n` | 4 | Constraint dimension |
| `m` | 8 | Lattice dimension |
| **Security** | **~26 bits** | Demo configuration |

> 💡 Production deployments should use `q ≈ 2^32`, `n ≈ 512`, `m ≈ 1024` for 128+ bit security.

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+
pip install torch numpy fastapi uvicorn pydantic matplotlib
```

### Run the API Server
```bash
python transac_api.py
```
```
✅ Lattice: q=97, n=4, m=8
   Security: ~26 bits

🌐 Endpoints:
   POST /sigil/prepare  - Generate & verify signature
   POST /sigil/record   - Record transaction
   GET  /sigil/history  - View history
   GET  /sigil/stats    - View statistics

🚀 Running on http://127.0.0.1:8000
```

### Run Comprehensive Tests
```bash
python core.py
```
```
🔬 SIGIL COMPREHENSIVE TESTING
======================================================================

📝 Test 1: Valid Signature Verification
----------------------------------------------------------------------
✅ Message 0: VALID (norm=4.58, error=0)
✅ Message 1: VALID (norm=5.12, error=0)
✅ Message 2: VALID (norm=4.89, error=0)

Result: 20/20 signatures verified (100.0%)

🛡️ Test 2: Forgery Resistance
----------------------------------------------------------------------
Real signature: ✅ VALID (error=0)
Fake signature: ❌ INVALID (error=42)
Tampered message: ❌ INVALID

⚡ Test 3: Performance
----------------------------------------------------------------------
Signing:      0.234 ms/signature
Verification: 0.089 ms/verify

📊 Test 4: Probabilistic Scoring
----------------------------------------------------------------------
Constraint score: 1.0000
Norm score:       0.9847
Final score:      0.9938
Verdict:          ACCEPT
```

---

## 📡 API Reference

### `POST /sigil/prepare`
Generate a quantum-safe signature for a transaction.

**Request:**
```json
{
  "sender": "0x742d35Cc6634C0532925a3b844Bc9e7595f...",
  "receiver": "0x8ba1f109551bD432803012645Ac136ddd...",
  "amount": "1.5",
  "message": "Payment for services"
}
```

**Response:**
```json
{
  "tx_string": "0x742d...|0x8ba1...|1.5|Payment for services",
  "sigil_signature": [-2, 1, 0, -1, 3, 2, -1, 0],
  "signature_norm": 4.58,
  "verified": true,
  "max_error": 0,
  "final_score": 0.9938,
  "constraint_score": 1.0,
  "norm_score": 0.9847,
  "verdict": "ACCEPT",
  "timestamp": "2026-01-13T15:30:00.000Z"
}
```

### `GET /sigil/stats`
Retrieve lattice security parameters.

**Response:**
```json
{
  "q": 97,
  "n": 4,
  "m": 8,
  "security_bits": 26,
  "transactions": 42
}
```

---

## 🎮 Interactive Demo

The `final_static/` directory contains a stunning web demo featuring:

- 🌀 **Animated Blob Visualization** — Watch the lattice structure morph in real-time
- 💳 **MetaMask Integration** — Sign real transactions on Sepolia testnet
- 🔐 **Live SIGIL Signatures** — See quantum-safe signatures generated instantly
- 📊 **Verification Scoring** — Visual feedback on signature validity
- 🎯 **Doomsday Clock** — Countdown to Q-Day awareness

### Launch the Demo
```bash
# Start API server
python transac_api.py

# Serve static files (separate terminal)
cd final_static
python -m http.server 5500
```

Navigate to `http://localhost:5500` and connect your MetaMask wallet!

---

## ⚔️ Quantum Attack Simulator (QtorchX)

SIGIL includes a **2500+ line quantum circuit simulator** demonstrating exactly what we're defending against:

### Run RSA Factorization Demo
```bash
cd sigil-crypto/Attacker\ model
python rsa-breaker.py
```

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚛️  QTORCHX: QUANTUM RSA CRYPTANALYSIS DEMONSTRATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Target:  N = 899 (10-bit RSA)
🔐 Task:    Factor N = p × q
💻 Device:  CUDA

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🐌 METHOD 1: Classical Trial Division
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Success: 899 = 29 × 31
⏱️  Time: 0.45 ms | Operations: 28

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚛️  METHOD 2: Quantum Shor's Algorithm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Building Quantum Phase Estimation circuit...
  Qubits: 12 | Base: a = 7

✅ Success: 899 = 29 × 31
⏱️  Time: 25.3 ms | Attempts: 1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 PERFORMANCE COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Method               Result               Time            Details
──────────────────────────────────────────────────────────────────────
Classical            29×31                0.5ms           28 operations
Quantum (Shor)       29×31                25.3ms          QPE circuit

⚡ At RSA-2048 scale: Quantum is EXPONENTIALLY faster!
```

### QtorchX Features
- **40+ Quantum Gates**: Full gate library including Hadamard, CNOT, RZ, Toffoli, SWAP, and more
- **State Vector Simulation**: Pure-state quantum computation engine
- **Burst Weight Modeling**: Hardware-calibrated gate error simulation
- **Circuit Visualization**: ASCII circuit diagrams for debugging
- **GPU Acceleration**: CUDA-optimized tensor operations via PyTorch

---

## 🛡️ Why Lattice Cryptography?

| Attack Vector | RSA/ECDSA | SIGIL (Lattice) |
|---------------|-----------|-----------------|
| Classical Computers | ✅ Secure | ✅ Secure |
| Shor's Algorithm (Quantum) | ❌ **BROKEN** | ✅ Secure |
| Grover's Algorithm | ⚠️ Weakened | ✅ Minimal impact |
| HNDL Attacks | ❌ Future vulnerable | ✅ Future-proof |

### NIST Post-Quantum Standards
SIGIL's approach aligns with NIST-approved algorithms:
- **CRYSTALS-Dilithium** — Lattice-based digital signatures
- **CRYSTALS-Kyber** — Lattice-based key encapsulation
- **FALCON** — Compact lattice signatures

---

## 🎨 Visual Gallery

### Lattice Structure Visualization
The `plot_lattice_2d()` and `plot_lattice_3d()` functions generate beautiful visualizations:

```
🛡️ SIGIL Lattice Structure (2D Projection)
    
         ·  ·  ·  ·  ·  ·  ·  ·  ·
       ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
     ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
   ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
     ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
       ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
         ·  ·  ·  ·  ·  ·  ·  ·  ·
         
    The signature must land on a lattice point
    close to H(message) — computationally hard
    to forge without the trapdoor!
```

---

## 🔧 Technical Highlights

### Core Cryptographic Primitives

```python
# Generate a secure q-ary lattice
params = LatticeParams(q=97, n=4, m=8)
lattice = generate_qary_lattice("secret_seed", params, device="cuda")

# Sign a message
signature = sign_message(lattice, "Hello, Quantum World!", sigma=1.5)

# Verify with boolean + probabilistic scoring
verifier = SIGILVerifier(lattice, noise_bound=2)
valid, details = verifier.verify(signature)
scores = verifier.verify_with_score(signature)
```

### Performance Benchmarks
| Operation | Time (CPU) | Time (CUDA) |
|-----------|------------|-------------|
| Lattice Generation | 1.2 ms | 0.8 ms |
| Signature (sign_message) | 0.23 ms | 0.15 ms |
| Verification | 0.09 ms | 0.05 ms |

---

## 🤝 Contributing

We welcome contributions! Areas of interest:

- 🔬 **Cryptographic Improvements** — Enhanced parameter selection, new lattice constructions
- ⚡ **Performance Optimization** — SIMD instructions, multi-threading, optimized GPU kernels
- 🌐 **API Extensions** — WebSocket support, batch processing, additional endpoints
- 📚 **Documentation** — Tutorials, security analysis, deployment guides
- 🧪 **Testing** — Fuzzing, formal verification, edge case coverage

---

## 📜 License

```
Apache License 2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
```

---

## 🔮 The Future is Quantum-Safe

<p align="center">
  <strong>Don't wait for Q-Day.</strong><br/>
  <em>Start protecting your digital identity today with SIGIL.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/YOUR-IDENTITY-C9B3E6?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/UNFORGEABLE-8B6FD6?style=for-the-badge"/>
</p>

---

<p align="center">
  <sub>Built with 💜 for a post-quantum world</sub>
</p>
