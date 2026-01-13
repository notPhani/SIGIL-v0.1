"""
SIGIL API
Vanilla FastAPI interface for lattice-based transaction signing.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import numpy as np

from core import (
    LatticeParams,
    generate_qary_lattice,
    sign_message,
    SIGILVerifier
)

# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="SIGIL",
    description="Lattice-based, post-quantum transaction signing demo",
    version="0.1.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Global lattice (demo)
# ------------------------------------------------------------------

PARAMS = LatticeParams(q=97, n=4, m=8)

LATTICE = generate_qary_lattice(
    seed="sigil_demo_key",
    params=PARAMS,
    device="cpu"
)

VERIFIER = SIGILVerifier(LATTICE, noise_bound=2)
TRANSACTIONS = []

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class PrepareRequest(BaseModel):
    sender: str
    receiver: str
    amount: str
    message: Optional[str] = "SIGIL transaction"


class PrepareResponse(BaseModel):
    tx_string: str
    sigil_signature: List[int]
    signature_norm: float
    verified: bool
    max_error: float
    # Scoring details
    final_score: float
    constraint_score: float
    norm_score: float
    verdict: str
    timestamp: str


class RecordRequest(BaseModel):
    tx_hash: str
    sender: str
    receiver: str
    amount: str
    message: str
    sigil_signature: List[int]
    signature_norm: float


class StatsResponse(BaseModel):
    q: int
    n: int
    m: int
    security_bits: int
    transactions: int

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def build_tx_string(sender: str, receiver: str, amount: str, message: str) -> str:
    return f"{sender}|{receiver}|{amount}|{message}"

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.post("/sigil/prepare", response_model=PrepareResponse)
def prepare_transaction(req: PrepareRequest):
    """
    Generate and verify SIGIL signature with scoring
    """
    tx_string = build_tx_string(
        req.sender,
        req.receiver,
        req.amount,
        req.message
    )

    print(f"\n{'='*70}")
    print(f"🔐 SIGIL SIGNATURE GENERATION")
    print(f"Sender:   {req.sender}")
    print(f"Receiver: {req.receiver}")
    print(f"Amount:   {req.amount}")
    print(f"Message:  {req.message}")

    # Generate signature
    sig = sign_message(
        lattice=LATTICE,
        message=tx_string,
        sigma=1.5
    )

    print(f"✅ Signature Generated (Norm: {sig.norm:.2f})")

    # Boolean verification
    valid, details = VERIFIER.verify(sig)
    
    # Scoring verification
    scores = VERIFIER.verify_with_score(sig)

    print(f"\n🔍 VERIFICATION RESULTS:")
    print(f"   Boolean: {'✅ VALID' if valid else '❌ INVALID'}")
    print(f"   Score:   {scores['final_score']:.4f}")
    print(f"   Verdict: {scores['verdict']}")
    print(f"   Max Error: {details['max_error']}")
    print(f"{'='*70}\n")

    return PrepareResponse(
        tx_string=tx_string,
        sigil_signature=sig.s.tolist(),
        signature_norm=float(sig.norm),
        verified=valid,
        max_error=float(details["max_error"]),
        # Scoring fields
        final_score=float(scores["final_score"]),
        constraint_score=float(scores["constraint_score"]),
        norm_score=float(scores["norm_score"]),
        verdict=scores["verdict"],
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/sigil/record")
def record_transaction(req: RecordRequest):
    """
    Record completed transaction
    """
    record = {
        **req.dict(),
        "timestamp": datetime.utcnow().isoformat(),
        "quantum_safe": True
    }
    TRANSACTIONS.append(record)
    
    print(f"\n{'='*70}")
    print(f"✅ TRANSACTION RECORDED")
    print(f"Hash:   {req.tx_hash[:20]}...")
    print(f"Amount: {req.amount}")
    print(f"Norm:   {req.signature_norm:.2f}")
    print(f"{'='*70}\n")
    
    return {"success": True, "receipt": record}


@app.get("/sigil/history")
def history():
    """
    Get transaction history
    """
    return {
        "success": True,
        "count": len(TRANSACTIONS),
        "transactions": TRANSACTIONS
    }


@app.get("/sigil/stats", response_model=StatsResponse)
def stats():
    """
    Get lattice statistics
    """
    return StatsResponse(
        q=PARAMS.q,
        n=PARAMS.n,
        m=PARAMS.m,
        security_bits=int(PARAMS.n * np.log2(PARAMS.q)),
        transactions=len(TRANSACTIONS)
    )


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print(f"\n✅ Lattice: q={PARAMS.q}, n={PARAMS.n}, m={PARAMS.m}")
    print(f"   Security: ~{int(PARAMS.n * np.log2(PARAMS.q))} bits")
    print(f"\n🌐 Endpoints:")
    print(f"   POST /sigil/prepare  - Generate & verify signature")
    print(f"   POST /sigil/record   - Record transaction")
    print(f"   GET  /sigil/history  - View history")
    print(f"   GET  /sigil/stats    - View statistics")
    print(f"\n🚀 Running on http://127.0.0.1:8000\n")
    
    uvicorn.run(
        "transac_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )
