import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader, random_split
from scripts.converter import converter
from scripts.dataset import MNISTDataset
from scripts.preprocess_custom import transformer_custom
from scripts.preprocess_for_conv import transformer_for_conv
from scripts.train_model import train_model
from src.models.CustomModel.CustomModel import CustomModel
from src.models.VGG.VGG import VGG
from src.models.ResNet.ResNet import ResNet
from utils.test import test

def main():
    print('-- Выбираем устройство на котором будем обучать --')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('-- Конвертируем данные --')
    test_dir = 'prepared_data/test'
    train_dir = 'prepared_data/train'

    train = pd.read_csv('data/train/fashion-mnist_train.csv')
    test_df_raw = pd.read_csv('data/test/fashion-mnist_test.csv')

    # converter(train, train_dir)
    # converter(test_df_raw, test_dir)

    print('-- Подготавливаем test_df и train_df датасеты --')
    
    # Выбираем трансформер в зависимости от модели!
    # Для Custom модели (MLP)
    # train_df = MNISTDataset('prepared_data/train', transformer_custom())
    # test_df = MNISTDataset('prepared_data/test', transformer_custom())
    
    # Для VGG и ResNet (CNN)
    train_df = MNISTDataset('prepared_data/train', transformer_for_conv())
    test_df = MNISTDataset('prepared_data/test', transformer_for_conv())
    print('-- Датасеты подготовлены --')

    print('-- Разбиваем на train_data и val_data выборки подготовленный train датасет --')
    train_data, val_data = random_split(train_df, [0.8, 0.2])
    print('-- Выборки подготовлены')

    batch_size = 32
    print(f'-- Разбиваем выборки на батчи, train_data перемешиваем, размер батча {batch_size} --')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_df, batch_size=batch_size, shuffle=False)
    print('-- Разбитие на батчи готово --')

    print('-- Инициализируем модель --')
    custom_model = CustomModel(784, 10).to(device)
    vgg_model = VGG().to(device)
    resnet_model = ResNet().to(device)
    print('-- Модель инициализирована --')

    print('-- Выбор функции потерь и оптимизатора --')
    loss = nn.CrossEntropyLoss()
    
    # Создаем ОПТИМИЗАТОРЫ (каждый для своей модели)
    custom_model_optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.001)
    vgg_model_optimizer = torch.optim.Adam(vgg_model.parameters(), lr=0.001)
    resnet_model_optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.001)

    # Создаем ПЛАНИРОВЩИКИ (каждый привязан к своему оптимизатору)
    custom_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(custom_model_optimizer, mode='min', factor=0.1, patience=5)
    vgg_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(vgg_model_optimizer, mode='min', factor=0.1, patience=5)
    resnet_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(resnet_model_optimizer, mode='min', factor=0.1, patience=5)
    print('-- Функция потерь и оптимизатор выбраны --')

    print('-- Обучаем модель --')
    # 1. Обучение CustomModel
    # train_model(custom_model, train_loader, val_loader, loss, device, custom_model_optimizer, custom_lr_scheduler)
    # 2. Обучение VGG
    # train_model(vgg_model, train_loader, val_loader, loss, device, vgg_model_optimizer, vgg_lr_scheduler)
    # 3. Обучение ResNet
    train_model(resnet_model, train_loader, val_loader, loss, device, resnet_model_optimizer, resnet_lr_scheduler)
    print('-- Модель обучена --')

    print('-- Тестируем модель --')
    # 1. Тест CustomModel
    # mean_test_loss, running_test_acc = test(custom_model, test_loader, loss, device) 
    # 2. Тест VGG
    # mean_test_loss, running_test_acc = test(vgg_model, test_loader, loss, device)
    # 3. Тест ResNet
    mean_test_loss, running_test_acc = test(resnet_model, test_loader, loss, device)

    print(f'test_loss={mean_test_loss:.4f}')
    print(f'test_acc={running_test_acc:.4f}')

if __name__ == "__main__":
    main()