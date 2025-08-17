import os, argparse, torch, numpy as np
from config import Config
from utils.ckpt import load_ckpt
from utils.plot import plot_density
from data.datasets import make_loaders
from models.hybrid_model import HybridQuantumModel

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, default=os.path.join(Config.data_dir, Config.dataset_name + ".npz"))
    ap.add_argument("--device", type=str, default=Config.device)
    ap.add_argument("--out-dir", type=str, default="eval_out")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    dl_tr, dl_va = make_loaders(args.data, batch_size=64)

    grid_points = dl_tr.dataset.X.shape[-1]
    model = HybridQuantumModel(grid_points,
                               n_qubits=Config.n_qubits,
                               q_layers=Config.q_layers,
                               embed_dim=Config.embed_dim,
                               mlp_hidden=Config.mlp_hidden,
                               shots=None).to(device)

    load_ckpt(args.ckpt, model=model, map_location=device)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    mse_list, mae_list = [], []

    with torch.no_grad():
        for i, (X, y) in enumerate(dl_va):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            mse = torch.mean((y_hat - y)**2, dim=1).cpu().numpy()
            mae = torch.mean(torch.abs(y_hat - y), dim=1).cpu().numpy()
            mse_list.extend(mse.tolist())
            mae_list.extend(mae.tolist())

            if i == 0:
                # Plot first item from batch
                t = y[0].cpu().numpy()
                p = y_hat[0].cpu().numpy()
                plot_density(t, p, os.path.join(args.out_dir, "example_density.png"))

    print(f"Validation MSE: {np.mean(mse_list):.4e}")
    print(f"Validation MAE: {np.mean(mae_list):.4e}")
    print(f"Saved example plot → {os.path.join(args.out_dir, 'example_density.png')}")

if __name__ == "__main__":
    main()
