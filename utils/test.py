import torch

EPOCHS = 50
test_loss = []
test_acc = []

def test(model, test_loader, loss, device):
    # Валидация модели
    model.eval()
    correct = 0
    total = 0
    # Запрещаем считать градиент
    with torch.no_grad():
        running_test_loss = []

        for x, targets in test_loader: # x - батчи, targets - ответы
            # (batch_size, 1, 28, 28) -> (batch_size, 784)
            x = x.reshape(-1, 28*28).to(device)  # Преобразуем батчи в вектор
            # (batch_size, int) -> (batch_size, 10), dtype = float32
            # По сути One-Hot Encoding
            targets = targets.reshape(-1).to(torch.int32) # Например tensor([[0], [3], [7]]) -> tensor([0, 3, 7])
            targets = torch.eye(10)[targets].to(device)
    
            # Прямой проход + расчет ошибки
            pred = model(x)
            batch_loss = loss(pred, targets)

            running_test_loss.append(batch_loss.item())

            correct += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
            total += targets.size(0)

        mean_test_loss = sum(running_test_loss) / len(running_test_loss)

        # Рассчет accuracy
        running_test_acc = correct / total

        test_loss.append(mean_test_loss)
        test_acc.append(running_test_acc)

    return mean_test_loss, running_test_acc

if __name__ == "__main__":
    test()    