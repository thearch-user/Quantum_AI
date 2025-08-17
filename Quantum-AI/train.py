import os, math, argparse, torch, numpy as np
from config import Config
from utils.seed import set_seed
from utils.timer import Timer
from utils.ckpt import save_ckpt
from utils.amp import AmpScaler
from data.datasets import make_loaders
from models.hybrid_model import HybridQuantumModel

def cosine_with_warmup(optimizer, warmup, total_steps, min_lr=1e-5):
    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = float(step - warmup) / float(max(1, total_steps - warmup))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--batch-size", type=int, default=Config.batch_size)
    ap.add_argument("--device", type=str, default=Config.device)
    ap.add_argument("--data", type=str, default=os.path.join(Config.data_dir, Config.dataset_name + ".npz"))
    ap.add_argument("--run-dir", type=str, default=Config.run_dir)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    ap.add_argument("--grad-clip", type=float, default=Config.grad_clip)
    ap.add_argument("--n-qubits", type=int, default=Config.n_qubits)
    ap.add_argument("--q-layers", type=int, default=Config.q_layers)
    ap.add_argument("--q-shots", type=int, default=0)  # 0→analytic
    ap.add_argument("--grid", type=int, default=Config.grid_points)
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(Config.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")

    dl_tr, dl_va = make_loaders(args.data, batch_size=args.batch_size)
    grid_points = dl_tr.dataset.X.shape[-1]

    model = HybridQuantumModel(grid_points=grid_points,
                               n_qubits=args.n_qubits,
                               q_layers=args.q_layers,
                               embed_dim=Config.embed_dim,
                               mlp_hidden=Config.mlp_hidden,
                               shots=None if args.q_shots==0 else args.q_shots).to(device)

    # Try torch.compile for extra speed (PyTorch 2+)
    try:
        model = torch.compile(model)  # type: ignore
    except Exception:
        pass

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * (len(dl_tr))
    sched = cosine_with_warmup(optim, warmup=Config.warmup_steps, total_steps=total_steps, min_lr=Config.cosine_min_lr)

    scaler = AmpScaler(enabled=(device.type == "cuda"))

    os.makedirs(args.run_dir, exist_ok=True)
    best_loss = float("inf")
    best_path = os.path.join(args.run_dir, "best.pt")

    def step(batch, train=True):
        model.train(mode=train)
        X, y = [t.to(device, non_blocking=True) for t in batch]
        with scaler.autocast():
            y_hat = model(X)                        # (B, G)
            loss = torch.nn.functional.mse_loss(y_hat, y)
        if train:
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optim)
            scaler.step(optim)  # harmless double-step prevention? (noop)
            scaler.scaler.update()
            sched.step()
        return loss.item()

    global_step = 0
    for epoch in range(args.epochs):
        # Train
        losses = []
        with Timer() as tr_t:
            for batch in dl_tr:
                l = step(batch, train=True)
                losses.append(l)
                global_step += 1
        train_loss = float(np.mean(losses))

        # Val
        model.eval()
        vlosses = []
        with torch.no_grad(), Timer() as va_t:
            for batch in dl_va:
                vlosses.append(step(batch, train=False))
        val_loss = float(np.mean(vlosses))

        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"train {train_loss:.4e} ({tr_t.dt:.2f}s) | "
              f"val {val_loss:.4e} ({va_t.dt:.2f}s)")

        if val_loss < best_loss:
            best_loss = val_loss
            save_ckpt(best_path, model, optim, sched, global_step, best=True)
            print(f"  ↳ New best: {best_loss:.4e} → {best_path}")

    print("Done.")

if __name__ == "__main__":
    main()
