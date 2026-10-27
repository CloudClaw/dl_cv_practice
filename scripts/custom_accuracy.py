def custom_accuracy(loader, pred, targets):
    true_answers = 0

    true_answers += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

    running_acc = true_answers / len(loader)

    return running_acc

if __name__ == "__main__":
    custom_accuracy()