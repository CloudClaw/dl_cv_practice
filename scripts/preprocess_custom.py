import torchvision.transforms.v2 as v2

def transformer_custom():
    return v2.Compose([
        v2.Normalize(mean=[0.5], std=[0.5])
    ])

if __name__ == "__main__":
    transformer_custom()