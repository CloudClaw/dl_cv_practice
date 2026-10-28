import torch

def train(model, train_loop, device, optimizer, loss, epoch, EPOCHS):
    model.train()
    running_train_loss = []
    correct = 0
    total = 0

    for x, targets in train_loop:
        x = x.to(device)
        
        # 2. Приводим targets к int64 (без One-Hot, так как используем CrossEntropyLoss)
        targets = targets.reshape(-1).to(torch.int64).to(device)

        pred = model(x)
        batch_loss = loss(pred, targets)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        running_train_loss.append(batch_loss.item())
        mean_train_loss = sum(running_train_loss) / len(running_train_loss)

        correct += (pred.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)

        train_loop.set_description(f'Epoch {epoch}/{EPOCHS}, train_loss={mean_train_loss:.4f}')

    running_train_acc = correct / total
    return mean_train_loss, running_train_acc