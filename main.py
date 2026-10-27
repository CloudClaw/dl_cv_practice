import pandas as pd

from torch.utils.data import DataLoader, random_split
from scripts.converter import converter
from scripts.dataset import MNISTDataset
from scripts.preprocess import transformer

def main():
    print('Конвертируем данные')
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

if __name__ == "__main__":
    main()