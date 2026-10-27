import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader, random_split
from scripts.converter import converter
from scripts.dataset import MNISTDataset
from scripts.preprocess import transformer
from scripts.train import train
from src.models.CustomModel import CustomModel

def main():
    print('-- Выбираем устройство на котором будем обучать --')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('-- Конвертируем данные --')
    test_dir = 'prepared_data/test'
    train_dir = 'prepared_data/train'

    train = pd.read_csv('data/train/fashion-mnist_train.csv')
    test = pd.read_csv('data/test/fashion-mnist_test.csv')

    # converter(train, train_dir)
    # converter(test, test_dir)

    print('-- Подготавливаем test_df и train_df датасеты --')
    train_df = MNISTDataset('prepared_data/train', transformer)
    test_df = MNISTDataset('prepared_data/test', transformer)
    print('-- Датасеты подготовлены --')

    print('-- Разбиваем на train_data и val_data выборки подготовленный train датасет --')
    train_data, val_data = random_split(train_df, [0.8, 0.2])
    print('-- Выборки подготовлены')

    batch_size = 16
    print(f'-- Разбиваем выборки на батчи, train_data перемешиваем, размер батча {batch_size} --')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_df, batch_size=batch_size, shuffle=False)
    print('-- Разбитие на батчи готово --')

    print('-- Инициализирауем модель --')
    model = CustomModel(784, 10).to(device)
    print('-- Модель инициализирована --')

    input = torch.randn([16, 784], dtype=torch.float32).to(device)
    out = model(input)
    out.shape

    print('-- Выбор функции потерь и оптимизатора --')
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print('-- Функция потерь и оптимизатор выбраны --')

    print('-- Обучаем модель --')
    train(model, train_loader, val_loader, loss, optimizer, device)

if __name__ == "__main__":
    main()