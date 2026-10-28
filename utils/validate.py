import torch

val_loss = []
val_acc = []

def validate(model, val_loader, loss, device, epoch, EPOCHS, train_loop):
    # Валидация модели
    model.eval()
    correct = 0
    total = 0
    # Запрещаем считать градиент
    with torch.no_grad():
        running_val_loss = []

        for x, targets in val_loader: # x - батчи, targets - ответы
            # (batch_size, 1, 28, 28) -> (batch_size, 784)
            x = x.reshape(-1, 28*28).to(device)  # Преобразуем батчи в вектор
            # (batch_size, int) -> (batch_size, 10), dtype = float32
            # По сути One-Hot Encoding
            targets = targets.reshape(-1).to(torch.int32) # Например tensor([[0], [3], [7]]) -> tensor([0, 3, 7])
            targets = torch.eye(10)[targets].to(device)
    
            # Прямой проход + расчет ошибки
            pred = model(x)
            batch_loss = loss(pred, targets)

            running_val_loss.append(batch_loss.item())

            correct += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
            total += targets.size(0)

            train_loop.set_description(f'Epoch {epoch}/{EPOCHS}, val_loss={batch_loss.item():.4f}')

        mean_val_loss = sum(running_val_loss) / len(running_val_loss)

        # Рассчет accuracy
        running_val_acc = correct / total

        val_loss.append(mean_val_loss)
        val_acc.append(running_val_acc)

    return mean_val_loss, running_val_acc

if __name__ == "__main__":
    validate()    