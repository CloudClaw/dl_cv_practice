import torch

def validate(model, val_loop, loss, device, epoch, EPOCHS):
    model.eval() # Включаем режим валидации/теста (отключает Dropout)
    correct = 0
    total = 0
    running_val_loss = []

    with torch.no_grad(): # Отключаем расчет градиентов (экономит память)
        for x, targets in val_loop:
            x = x.to(device)
            targets = targets.reshape(-1).to(torch.int64).to(device)

            pred = model(x)
            batch_loss = loss(pred, targets)

            running_val_loss.append(batch_loss.item())

            correct += (pred.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

            val_loop.set_description(f'Epoch {epoch}/{EPOCHS}, val_loss={batch_loss.item():.4f}')

        mean_val_loss = sum(running_val_loss) / len(running_val_loss)
        running_val_acc = correct / total

    return mean_val_loss, running_val_acc