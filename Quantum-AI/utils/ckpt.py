import os, torch

def save_ckpt(path, model, optimizer, scheduler, step, best=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "best": best
    }, path)

def load_ckpt(path, model=None, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    if model: model.load_state_dict(ckpt["model"])
    if optimizer and ckpt.get("optimizer"): optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
