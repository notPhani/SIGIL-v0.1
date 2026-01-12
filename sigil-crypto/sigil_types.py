from typing import TypedDict
import torch

class Lattice(TypedDict):
    basis: torch.Tensor       # A matrix (n×m) on GPU
    dimension: int            # n
    short_dim: int            # m
    seed: str                 # user_id
    device: str               

class Signature(TypedDict):
    t: torch.Tensor           # Signature vector
    lattice_id: str
    identity: str
    timestamp: float