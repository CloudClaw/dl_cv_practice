import os

from PIL import Image
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
  def __init__(self, path, transform=None):
    self.path = path
    self.transform = transform

    self.len_dataset = 0
    self.data_list = []

    for path_dir, dir_list, file_list in os.walk(path):
      if path_dir == path:
          self.classes = sorted(dir_list)
          self.class_to_idx = {
              cls_name: i for i, cls_name in enumerate(self.classes)
          }
          continue

      cls = os.path.basename(path_dir)

      for name_file in file_list:
          file_path = os.path.join(path_dir, name_file)
          self.data_list.append((file_path, self.class_to_idx[cls]))

      self.len_dataset += len(file_list)

  def __len__(self):
    return self.len_dataset

  def __getitem__(self, index):
    file_path, target = self.data_list[index]
    image = Image.open(file_path)

    if self.transform is not None:
      image = self.transform(image)

    return image, target