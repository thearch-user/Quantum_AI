# Quantum-AI (Hybrid) Starter

A high-performance starter that:
- Generates supervised data by solving a 1D Schrödinger equation (finite-difference).
- Trains a hybrid quantum-classical model (PennyLane + PyTorch).
- Uses fast PennyLane backends, AMP, gradient clipping, cosine LR schedule, and checkpoints.

## Quickstart
```bash
pip install -e .
# or: pip install -r requirements.txt
python -m data.generate_dataset --n-samples 6000 --grid 256 --train-split 0.8
python train.py --epochs 30 --batch-size 128 --device cuda
python eval.py --ckpt runs/best.pt
