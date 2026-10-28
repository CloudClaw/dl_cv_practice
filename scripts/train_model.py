import torch

from tqdm import tqdm
from utils.train import train
from utils.validate import validate
from utils.test import test

EPOCHS = 10
lr_list = []
best_loss = None
idle = 0 # Число итерация без улучшений модели

def train_model(model, train_loader, val_loader, loss, optimizer, lr_sheduler):
    global best_loss, idle

    for epoch in range(EPOCHS):
        train_loop = tqdm(train_loader, leave=False) # Добавляем прогресс бар
        val_loop = tqdm(val_loader, leave=False) 
        
        mean_train_loss, running_train_acc = train(
            model, train_loop, optimizer, loss, epoch + 1, EPOCHS)

        mean_val_loss, running_val_acc = validate(
            model, val_loop, loss, epoch + 1, EPOCHS)      

        lr_sheduler.step(mean_val_loss)
        lr = optimizer.param_groups[0]['lr'] # Значение скорости обучения
        lr_list.append(lr)

        print(f'Epoch {epoch}/{EPOCHS}')
        print(f'train_loss={mean_train_loss:.4f}')
        print(f'train_acc={running_train_acc:.4f}')
        print(f'val_loss={mean_val_loss:.4f}')
        print(f'val_acc={running_val_acc:.4f}')
        print(f'lr={lr: .4f}')

        if best_loss is None:
            best_loss = mean_val_loss

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            idle = 0
            # Сохраняем модель в зависимости от ее типа (чтобы не перезаписать)
            model_name = model.__class__.__name__
            torch.save(model.state_dict(), f'src/models/{model_name}/best_model.pt')
        else:
            idle += 1

        if idle >= 10:
            print(f'Обучение модели остановлено на {epoch + 1} эпохе.')
            break

if __name__ == "__main__":
    train_model()