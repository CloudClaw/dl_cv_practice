Классификация Fashion-MNIST: Сравнение MLP, VGG и ResNet

Данный проект представляет собой полноценный пайплайн по обучению и сравнению трех различных архитектур нейронных сетей (полносвязной MLP, сверточных VGG19 и ResNet50) на популярном датасете изображений одежды Fashion-MNIST.

Структура проекта

dl_cv_practice/
├── data/                      # Исходные CSV-файлы (создается вручную)
│   ├── train/                 # fashion-mnist_train.csv
│   └── test/                  # fashion-mnist_test.csv
├── prepared_data/             # Сконвертированные PNG-изображения (создается автоматически)
│   ├── train/
│   └── test/
├── scripts/
│   ├── converter.py           # Скрипт конвертации CSV в изображения PNG
│   ├── dataset.py             # Кастомный класс Dataset (PyTorch)
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
- Поместите скачанные архивы (или извлеченные CSV-файлы fashion-mnist_train.csv и fashion-mnist_test.csv) в соответствующие папки:
- data/train/fashion-mnist_train.csv
- data/test/fashion-mnist_test.csv
2. Установка зависимостей
uv sync
3. Конвертация данных
Необходимо раскомментировать converter, конвертация происходит 1 раз, это создаст папку prepared_data, внутри которой будут лежать изображения, разбитые по классам (например, prepared_data/train/T-shirt_top/img_00001.png).
4. В файле main.py необходимо выбрать модель для обучения
5. Запуск проекта
uv run main.py