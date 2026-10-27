import torch

def train_cycle(model, loader, device, loss):
    for x, targets in loader:     # x - батчи, targets - ответы
        # (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.reshape(-1, 28*28).to(device)  # Преобразуем батчи в вектор
        # (batch_size, int) -> (batch_size, 10), dtype = float32
        # По сути One-Hot Encoding
        targets = targets.reshape(-1).to(torch.int32) # Например tensor([[0], [3], [7]]) -> tensor([0, 3, 7])
        targets = torch.eye(10)[targets].to(device)

        # Прямой проход + расчет ошибки
        pred = model(x)
        loss = loss(pred, targets)

    return pred, targets, loss

if __name__ == "__main__":
    train_cycle()