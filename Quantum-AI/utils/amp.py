import torch

class AmpScaler:
    def __init__(self, enabled: bool):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)

    def autocast(self):
        return torch.cuda.amp.autocast(enabled=self.enabled)

    def step(self, optimizer):
        self.scaler.step(optimizer)
        self.scaler.update()

    def scale(self, loss):
        return self.scaler.scale(loss) if self.enabled else loss
