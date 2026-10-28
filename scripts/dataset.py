import torch
import pandas as pd

from torch.utils.data import Dataset

class MNISTDataset(Dataset):
  def __init__(self, csv_file, transform=None):
        # Читаем весь CSV в память (это быстро)
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Отделяем метки от пикселей
        self.labels = self.data['label'].values
        self.pixels = self.data.drop('label', axis=1).values
      
  def __len__(self):
      return len(self.data)
  
  def __getitem__(self, idx):
      # 1. Берем пиксели и метку
      pixels = self.pixels[idx]
      label = self.labels[idx]
      
      # 2. Превращаем в тензор float32 и нормализуем в [0, 1]
      # shape: (784,)
      image = torch.tensor(pixels, dtype=torch.float32) / 255.0
      
      # 3. Меняем форму на (1, 28, 28) для картинок
      image = image.view(1, 28, 28)
      
      # 4. Применяем трансформации (например, Resize до 64x64 и Grayscale в 3 канала)
      if self.transform:
          image = self.transform(image)
          
      return image, label