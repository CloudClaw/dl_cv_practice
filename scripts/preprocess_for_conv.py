import torch
import torchvision.transforms.v2 as v2

def transformer_for_conv():
    return v2.Compose([
        # 1. Ресайз до 224x224 (VGG ожидает минимум 224)
        v2.Resize((224, 224)),
        # 2. Если картинка 1-канальная (PNG), конвертируем в 3 канала
        v2.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        # 3. Превращаем в тензор (в v2 ToTensor работает как ToImage + ToDtype)
        v2.ToTensor(),
        # 4. Нормализация под ImageNet (так как мы используем предобученные веса)
        v2.Normalize(mean=[0.5], std=[0.5])
    ])

if __name__ == "__main__":
    transformer_for_conv()