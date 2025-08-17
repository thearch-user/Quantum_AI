from dataclasses import dataclass

@dataclass
class Config:
    # data
    data_dir: str = "cache"
    dataset_name: str = "schrod_1d_npz"
    grid_points: int = 256
    train_split: float = 0.8

    # training
    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    cosine_min_lr: float = 1e-5
    warmup_steps: int = 200

    # model
    n_qubits: int = 6
    q_layers: int = 2
    q_shots: int | None = None     # None = analytic
    embed_dim: int = 64
    mlp_hidden: int = 128
    out_dim: int = 1               # predict density at each x; head expands later

    # system
    device: str = "cuda"           # "cuda" or "cpu"
    seed: int = 42
    run_dir: str = "runs"