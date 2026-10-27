import os
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

def make_split():
  current_dir = os.path.dirname(os.path.abspath(__file__))

  # Поднимаемся на один уровень вверх (из scripts в dl_cv_practice)
  project_root = os.path.dirname(current_dir)
  
  # Собираем полный путь к файлу
  file_path = os.path.join(project_root, 'data', 'train', 'fashion-mnist_train.csv')
  
  # Загружаем файл
  print('Загружаем файлы')
  df = pd.read_csv(file_path)
  print('Файлы загружены')
  print(df.head())
  print(df.dtypes)

  # Разбиваем на train и test
  print('Разбиваем на train и test')
  train_df, val_df = train_test_split(
    df,
    test_size=0.15,
    random_state=SEED
  )
  print(train_df.shape)
  print(val_df.shape)

  # Сохраняем новые файлы
  print('Сохраняем полученные файлы')
  save_train_file_path = os.path.join(project_root, 'prepared_data', 'train', 'train.csv')
  save_val_file_path = os.path.join(project_root, 'prepared_data', 'test', 'val.csv')

  train_df.to_csv(save_train_file_path)
  val_df.to_csv(save_val_file_path)
  print('Файлы сохранены')

  return train_df, val_df

if __name__ == '__main__':
  make_split()