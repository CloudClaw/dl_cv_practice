import os
import numpy as np

from PIL import Image

class_names = {
    0: "T-shirt_top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle_boot"
}

def converter(df, base_dir):
  os.makedirs(base_dir, exist_ok=True)

  for index, row in df.iterrows():
    # Получаем метку (label)
    label = row['label']
    class_name = class_names[label]
    
    # Создаем папку для класса
    class_dir = os.path.join(base_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Берем 784 значения пикселей (все столбцы, кроме 'label')
    pixels = row.drop('label').values.astype(np.uint8)
    
    # Преобразуем плоский массив (784) в матрицу 28x28
    image_array = pixels.reshape(28, 28)
    
    # Создаем изображение и сохраняем его в нужную папку
    img = Image.fromarray(image_array)
    
    # Формируем имя файла (например, img_00001.png)
    img_name = f"img_{index:05d}.png"
    img_path = os.path.join(class_dir, img_name)
    img.save(img_path)

  print("Конвертация завершена!")

if __name__ == "__main__":
    converter()