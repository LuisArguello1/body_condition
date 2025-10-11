import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DogBodyConditionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.conditions = []

        # Recorremos carpetas de condición corporal
        for condition in os.listdir(root_dir):
            condition_path = os.path.join(root_dir, condition)
            if os.path.isdir(condition_path):
                if condition not in self.conditions:
                    self.conditions.append(condition)

                for img_name in os.listdir(condition_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(condition_path, img_name)
                        self.samples.append((img_path, condition))

        # Mapear etiquetas a índices
        self.condition_to_idx = {condition: i for i, condition in enumerate(sorted(self.conditions))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, condition = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        condition_label = self.condition_to_idx[condition]

        return image, condition_label


# --------------------------
# USO DEL DATASET
# --------------------------

DATA_DIR = "dataset/train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = DogBodyConditionDataset(DATA_DIR, transform=transform)

print("Condiciones corporales detectadas:", dataset.conditions)
print("Mapeo de índices:", dataset.condition_to_idx)

img, condition_label = dataset[0]
print(f"Ejemplo -> Imagen tamaño: {img.shape}, Condición corporal (índice): {condition_label}, Nombre: {list(dataset.condition_to_idx.keys())[list(dataset.condition_to_idx.values()).index(condition_label)]}")
print(f"Total de imágenes en el dataset: {len(dataset)}")
