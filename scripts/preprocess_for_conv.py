import torch
import torchvision.transforms.v2 as v2

def transformer_for_conv():
    return v2.Compose([
        v2.Resize((64, 64)),
        # 2. Если картинка 1-канальная (PNG), конвертируем в 3 канала
        v2.Grayscale(num_output_channels=3),
        # 3. Превращаем в тензор (в v2 ToTensor работает как ToImage + ToDtype)
        v2.ToTensor(),
        # 4. Нормализация под ImageNet (так как мы используем предобученные веса)
        v2.Normalize(mean=[0.5], std=[0.5])
    ])

if __name__ == "__main__":
    transformer_for_conv()