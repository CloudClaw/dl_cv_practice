
import torch

from tqdm import tqdm
from scripts.custom_accuracy import custom_accuracy

EPOCHS = 50

train_loss = []
train_acc = []
val_loss = []
val_acc = []

def train_model(model, train_loader, val_loader, loss, optimizer, device):
    for epoch in range(EPOCHS):
        # Режим тренировки модели
        model.train()

        running_train_loss = []

        train_loop = tqdm(train_loader, leave=False) # Добавляем прогресс бар

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

            train_loop.set_description(f'Epoch {epoch}/{EPOCHS}, train_loss={mean_train_loss:.4f}')

        # Рассчет accuracy
        running_train_acc = custom_accuracy(train_loader, pred, targets)

        train_loss.append(mean_train_loss)
        train_acc.append(running_train_acc)

        # Валидация модели
        model.eval()

        running_val_loss = []

        # Запрещаем считать градиент
        with torch.no_grad():
            running_val_loss.append(loss.item())
            mean_val_loss = sum(running_val_loss) / len(running_val_loss)
            true_answers += (pred.argmax(dim=1) == targets.argmax(dim=1)).sum().item()

            train_loop.set_description(f'Epoch {epoch}/{EPOCHS}, val_loss={mean_val_loss:.4f}')

            for x, targets in val_loader:     # x - батчи, targets - ответы
                # (batch_size, 1, 28, 28) -> (batch_size, 784)
                x = x.reshape(-1, 28*28).to(device)  # Преобразуем батчи в вектор
                # (batch_size, int) -> (batch_size, 10), dtype = float32
                # По сути One-Hot Encoding
                targets = targets.reshape(-1).to(torch.int32) # Например tensor([[0], [3], [7]]) -> tensor([0, 3, 7])
                targets = torch.eye(10)[targets].to(device)
        
                # Прямой проход + расчет ошибки
                pred = model(x)
                batch_loss = loss(pred, targets)

            # Рассчет accuracy
            running_val_acc = custom_accuracy(val_loader, pred, targets)

            val_loss.append(mean_val_loss)
            val_acc.append(running_val_acc)

            print(f'Epoch {epoch}/{EPOCHS}, train_loss={mean_train_loss:.4f}, train_acc={running_train_acc:.4f}, val_loss={mean_val_loss:.4f}, val_acc={running_val_acc:.4f}')

if __name__ == "__main__":
    train_model()