from abc import abstractmethod, ABC

import tqdm
import torch


class Trainer(ABC):

    @abstractmethod
    def train(self, train_loader):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

class SupervisedTrainer(Trainer):

    def __init__(self, network, optimizer, criterion, metrics={}, device="cpu"):
        self.network   = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics   = metrics
        self.device    = device
        self.network.to(device)
        self.criterion.to(device)

    def train(self, train_loader, epochs, val_loader=None, initial_epoch=1):
        for epoch in range(initial_epoch, initial_epoch + epochs):
            train_loss = 0
            pbar       = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")

            if val_loader:
                results = self.evaluate(val_loader, prefix="val")
            else:
                results = {}

            self.network.train()

            for i, batch in enumerate(pbar, 1):
                self.optimizer.zero_grad()

                inputs  = batch["input"].to(self.device)
                labels  = batch["label"].to(self.device)
                outputs = self.network(inputs)
                loss    = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                results["train_loss"] = train_loss / i
                pbar.set_postfix(results)

    def evaluate(self, loader, prefix=None):
        prefix  = f"{prefix}_" if prefix is not None else ""
        results = {f"{prefix}{key}": 0 for key in self.metrics}
        results[f"{prefix}loss"] = 0
        self.network.eval()

        with torch.no_grad():
            labels  = []
            outputs = []

            for batch in loader:
                inputs  = batch["input"].to(self.device)
                labels.append(batch["label"].to(self.device))
                outputs.append(self.network(inputs))

            labels  = torch.cat(labels)
            outputs = torch.cat(outputs)
            results = {f"{prefix}loss": self.criterion(outputs, labels).item()}
            results.update({
                f"{prefix}{key}": metric(outputs, labels).item()
                for key, metric in self.metrics.items()
            })

        return results

    def save(self, path):
        state_dict = self.network.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        self.network.load_state_dict(state_dict)