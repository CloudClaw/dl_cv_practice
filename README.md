Классификация Fashion-MNIST: Сравнение MLP, VGG и ResNet

Данный проект представляет собой полноценный пайплайн по обучению и сравнению трех различных архитектур нейронных сетей (полносвязной MLP, сверточных VGG19 и ResNet50) на популярном датасете изображений одежды Fashion-MNIST.

Структура проекта

dl_cv_practice/
├── data/                      # Исходные CSV-файлы (создается вручную)
│   ├── train/                 # fashion-mnist_train.csv
│   └── test/                  # fashion-mnist_test.csv
├── scripts/
│   ├── csv_dataset.py         # Кастомный класс Dataset для чтения CSV напрямую
│   ├── preprocess_custom.py   # Трансформации для MLP модели
│   ├── preprocess_for_conv.py # Трансформации для CNN моделей (VGG/ResNet)
│   └── train_model.py         # Главный цикл обучения с Early Stopping
├── src/
│   └── models/
│       ├── CustomModel/       # Полносвязная сеть (MLP)
│       ├── VGG/               # Сверточная сеть VGG19 (Transfer Learning)
│       └── ResNet/            # Сверточная сеть ResNet50 (Transfer Learning)
├── utils/
│   ├── train.py               # Функция шага обучения (train)
│   ├── validate.py            # Функция валидации (validate)
│   └── test.py                # Функция тестирования (test)
├── main.py                    # Точка входа в программу (запуск обучения)
├── pyproject.toml             # Зависимости проекта (uv)
└── README.md

Установка и запуск

1. Подготовка данных
- Скачайте датасет с Kaggle: Fashion-MNIST.
- В корне проекта создайте папку data.
- Внутри data создайте две папки: test и train.
- Поместите скачанные CSV-файлы fashion-mnist_train.csv и fashion-mnist_test.csv в соответствующие папки:
- data/train/fashion-mnist_train.csv
- data/test/fashion-mnist_test.csv
2. Установка зависимостей
uv sync
3. Запуск проекта
uv run main.py

==================================================
ИТОГОВЫЕ РЕЗУЛЬТАТЫ:
            Model  Test Loss  Test Acc
      CustomModel   0.376297    0.8645
            VGG19   0.408286    0.8598
         ResNet50   0.203951    0.9271
==================================================

Вывод:
ResNet выдал наименьшую ошибку за 5 эпох, получив максимальную точность по метрике accuracy (использовал ее, так как данные сбалансированы). Получается модель с точностью почти 93 процента указала верные классы картинкам.