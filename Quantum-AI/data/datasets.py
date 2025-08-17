import os, torch, numpy as np
from torch.utils.data import Dataset, DataLoader

class Schrodinger1D(Dataset):
    def __init__(self, path, split="train"):
        z = np.load(path)
        self.X = z["X"]   # (N, 2, grid)
        self.Y = z["Y"]   # (N, grid)
        tr_idx, va_idx = z["tr_idx"], z["va_idx"]
        self.ids = tr_idx if split == "train" else va_idx

        # Standardize features across dataset for stability
        xvals = self.X[:,0,:]  # all x are same grid; keep anyway
        vvals = self.X[:,1,:]
        self.x_mean = xvals.mean()
        self.x_std  = xvals.std() + 1e-6
        self.v_mean = vvals.mean()
        self.v_std  = vvals.std() + 1e-6

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        j = self.ids[i]
        feats = self.X[j]  # (2, grid)
        x = (feats[0] - self.x_mean) / self.x_std
        v = (feats[1] - self.v_mean) / self.v_std
        X = np.stack([x, v], axis=0)      # (2, grid)
        y = self.Y[j]                     # (grid,)
        return torch.from_numpy(X).float(), torch.from_numpy(y).float()

def make_loaders(path, batch_size=128, num_workers=2, pin_memory=True):
    ds_tr = Schrodinger1D(path, "train")
    ds_va = Schrodinger1D(path, "val")
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory,
                       drop_last=True, persistent_workers=num_workers>0)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory)
    return dl_tr, dl_va
