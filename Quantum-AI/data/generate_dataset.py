import os, argparse, numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

def schrodinger_1d(grid_n=256, potential="well", depth=50.0, seed=0):
    """
    Returns (x, density) for the ground state of a 1D Hamiltonian.
    Units are arbitrary; we set ħ^2/2m = 1 and x ∈ [0,1].
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, grid_n)
    dx = x[1] - x[0]

    # Base potentials
    if potential == "well":
        V = np.zeros_like(x)
        # add bumps to create variety
        for _ in range(3):
            c = rng.uniform(0.2, 0.8)
            w = rng.uniform(0.03, 0.12)
            a = rng.uniform(-depth, depth)
            V += a * np.exp(-0.5 * ((x - c)/w)**2)
        V = V - V.min()  # shift to non-negative
    elif potential == "harmonic":
        k = rng.uniform(10.0, 60.0)
        center = rng.uniform(0.3, 0.7)
        V = 0.5 * k * (x - center)**2
    else:
        raise ValueError("unknown potential")

    # Kinetic (finite difference, Dirichlet BC)
    main = np.full(grid_n, 2.0)
    off  = np.full(grid_n - 1, -1.0)
    T = diags([off, main, off], [-1, 0, 1]) / (dx*dx)

    # Hamiltonian H = T + diag(V)
    H = T + diags(V, 0)

    # Compute few lowest eigenpairs; take ground state
    # 'SM' — smallest magnitude; or use 'SA' for smallest algebraic
    vals, vecs = eigsh(H, k=3, which="SA", maxiter=2000)
    idx = np.argmin(vals)
    psi = vecs[:, idx]
    # Normalize
    psi = psi / np.linalg.norm(psi)
    density = psi**2
    density = density / (density.sum() * dx)  # ensure integrates to 1
    return x.astype(np.float32), V.astype(np.float32), density.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=6000)
    ap.add_argument("--grid", type=int, default=256)
    ap.add_argument("--train-split", type=float, default=0.8)
    ap.add_argument("--out-dir", type=str, default="cache")
    ap.add_argument("--name", type=str, default="schrod_1d_npz")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_list = []
    V_list = []
    Y_list = []

    for i in range(args.n_samples):
        pot = "well" if i % 2 == 0 else "harmonic"
        x, V, dens = schrodinger_1d(grid_n=args.grid, potential=pot, seed=i)
        # Features: [x, V(x)] per grid point; labels: density(x)
        feats = np.stack([x, V], axis=0)   # shape (2, grid)
        X_list.append(feats)
        V_list.append(V)                   # optional save
        Y_list.append(dens)

    X = np.stack(X_list, axis=0)  # (N, 2, grid)
    Y = np.stack(Y_list, axis=0)  # (N, grid)

    # Split
    N = X.shape[0]
    idx = np.arange(N)
    np.random.default_rng(123).shuffle(idx)
    split = int(N * args.train_split)
    tr_idx, va_idx = idx[:split], idx[split:]

    np.savez_compressed(
        os.path.join(args.out_dir, f"{args.name}.npz"),
        X=X, Y=Y, tr_idx=tr_idx, va_idx=va_idx
    )
    print(f"Saved {X.shape[0]} samples → {os.path.join(args.out_dir, args.name+'.npz')}")

if __name__ == "__main__":
    main()
