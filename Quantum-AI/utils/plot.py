import os
import numpy as np
import matplotlib.pyplot as plt

def plot_density(target, pred, out_path, title="Density (target vs pred)"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x = np.linspace(0, 1, len(target))
    plt.figure()
    plt.plot(x, target, label="target")
    plt.plot(x, pred, label="pred", linestyle="--")
    plt.legend()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("probability density")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
