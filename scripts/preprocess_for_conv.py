import torchvision.transforms.v2 as v2

def transformer_for_conv():
    return v2.Compose([
        # 1. Ресайз до 64x64 (работает с тензорами в v2)
        v2.Resize((64, 64)),
        # 2. Превращаем 1 канал в 3 канала (RGB)
        v2.Grayscale(num_output_channels=3),
        # 3. Нормализация
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

if __name__ == "__main__":
    transformer_for_conv()