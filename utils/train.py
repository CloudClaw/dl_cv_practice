import torch

from tqdm import tqdm

train_loss = []
train_acc = []

def train(model, train_loader, device, optimizer, loss, epoch, EPOCHS, train_loop):
    # Режим тренировки модели
    model.train()
    running_train_loss = []
    correct = 0
    total = 0

    for x, targets in train_loader: # x - батчи, targets - ответы
        # (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = x.reshape(-1, 28*28).to(device)  # Преобразуем батчи в вектор
        # (batch_size, int) -> (batch_size, 10), dtype = float32
        # По сути One-Hot Encoding
        targets = targets.reshape(-1).to(torch.int32) # Например tensor([[0], [3], [7]]) -> tensor([0, 3, 7])
        targets = torch.eye(10)[targets].to(device)

        # Прямой проход + расчет ошибки
        pred = model(x)
        batch_loss = loss(pred, targets)

        # Обратный проход
        optimizer.zero_grad()
        batch_loss.backward()

        # Шаг оптимизации
        optimizer.step()

        running_train_loss.append(batch_loss.item())
        mean_train_loss = sum(running_train_loss) / len(running_train_loss)

        correct += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
        total += targets.size(0)

        train_loop.set_description(f'Epoch {epoch}/{EPOCHS}, train_loss={mean_train_loss:.4f}')

    # Рассчет accuracy
    running_train_acc = correct / total

    train_loss.append(mean_train_loss)
    train_acc.append(running_train_acc)

    return mean_train_loss, running_train_acc, train_loop

if __name__ == '__main__':
    train()