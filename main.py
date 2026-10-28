import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader, random_split
from scripts.dataset import MNISTDataset
from scripts.preprocess_custom import transformer_custom
from scripts.preprocess_for_conv import transformer_for_conv
from scripts.train_model import train_model
from src.models.CustomModel.CustomModel import CustomModel
from src.models.VGG.VGG import VGG
from src.models.ResNet.ResNet import ResNet
from utils.test import test

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'-- Устройство: {device} --')

    # ==========================================
    # 1. ПОДГОТОВКА ДАННЫХ
    # ==========================================
    print('-- Подготавливаем данные --')
    batch_size = 32

    # А) Датасеты для CustomModel (MLP) - плоские, 1 канал, 28x28
    train_df_custom = MNISTDataset('data/train/fashion-mnist_train.csv', transformer_custom())
    test_df_custom = MNISTDataset('data/test/fashion-mnist_test.csv', transformer_custom())
    train_data_c, val_data_c = random_split(train_df_custom, [0.8, 0.2])

    train_loader_c = DataLoader(train_data_c, batch_size=batch_size, shuffle=True)
    val_loader_c = DataLoader(val_data_c, batch_size=batch_size, shuffle=False)
    test_loader_c = DataLoader(test_df_custom, batch_size=batch_size, shuffle=False)

    # Б) Датасеты для VGG и ResNet (CNN) - 3 канала, 64x64
    train_df_conv = MNISTDataset('data/train/fashion-mnist_train.csv', transformer_for_conv())
    test_df_conv = MNISTDataset('data/test/fashion-mnist_test.csv', transformer_for_conv())
    train_data_conv, val_data_conv = random_split(train_df_conv, [0.8, 0.2])
    
    train_loader_conv = DataLoader(train_data_conv, batch_size=batch_size, shuffle=True)
    val_loader_conv = DataLoader(val_data_conv, batch_size=batch_size, shuffle=False)
    test_loader_conv = DataLoader(test_df_conv, batch_size=batch_size, shuffle=False)
    print('-- Данные готовы --')

    # ==========================================
    # 2. ОБУЧЕНИЕ МОДЕЛЕЙ
    # ==========================================
    loss = nn.CrossEntropyLoss()
    results = [] # Сюда будем собирать метрики для таблицы

    # --- Модель 1: CustomModel (MLP) ---
    print('\n=== Обучение CustomModel (MLP) ===')
    custom_model = CustomModel(784, 10).to(device)
    opt_custom = torch.optim.Adam(custom_model.parameters(), lr=0.001)
    sched_custom = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_custom, mode='min', patience=3)
    
    train_model(custom_model, train_loader_c, val_loader_c, loss, device, opt_custom, sched_custom)
    test_loss, test_acc = test(custom_model, test_loader_c, loss, device)
    results.append({'Model': 'CustomModel (MLP)', 'Test Loss': test_loss, 'Test Acc': test_acc})

    # --- Модель 2: VGG ---
    print('\n=== Обучение VGG19 ===')
    vgg_model = VGG().to(device)
    opt_vgg = torch.optim.Adam(vgg_model.parameters(), lr=0.001)
    sched_vgg = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_vgg, mode='min', patience=3)
    
    train_model(vgg_model, train_loader_conv, val_loader_conv, loss, device, opt_vgg, sched_vgg)
    test_loss, test_acc = test(vgg_model, test_loader_conv, loss, device)
    results.append({'Model': 'VGG19', 'Test Loss': test_loss, 'Test Acc': test_acc})

    # --- Модель 3: ResNet ---
    print('\n=== Обучение ResNet50 ===')
    resnet_model = ResNet().to(device)
    opt_resnet = torch.optim.Adam(resnet_model.parameters(), lr=0.001)
    sched_resnet = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_resnet, mode='min', patience=3)
    
    train_model(resnet_model, train_loader_conv, val_loader_conv, loss, device, opt_resnet, sched_resnet)
    test_loss, test_acc = test(resnet_model, test_loader_conv, loss, device)
    results.append({'Model': 'ResNet50', 'Test Loss': test_loss, 'Test Acc': test_acc})

    # ==========================================
    # 3. ВЫВОД РЕЗУЛЬТАТОВ
    # ==========================================
    print('\n' + '='*50)
    print('ИТОГОВЫЕ РЕЗУЛЬТАТЫ:')
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print('='*50)

if __name__ == "__main__":
    main()