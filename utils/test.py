import torch

def test(model, test_loader, loss, device):
    model.eval()
    correct = 0
    total = 0
    running_test_loss = []

    with torch.no_grad():
        for x, targets in test_loader:
            x = x.to(device)
            targets = targets.reshape(-1).to(torch.int64).to(device)

            pred = model(x)
            batch_loss = loss(pred, targets)

            running_test_loss.append(batch_loss.item())

            correct += (pred.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

        mean_test_loss = sum(running_test_loss) / len(running_test_loss)
        running_test_acc = correct / total

    return mean_test_loss, running_test_acc