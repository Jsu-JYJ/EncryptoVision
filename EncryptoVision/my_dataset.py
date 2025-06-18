from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def add_gaussian_noise(signal, mean, std):
    w = 1
    noise = torch.normal(mean, std, signal.size())
    noisy_signal = signal + noise * w
    return noisy_signal


class MyDataSet(Dataset):
    """自定义数据集"""
    def __init__(self, images_path: list, images_class: list, data_type: str, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        self.data_type = data_type

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        label = self.images_class[item]
        img = Image.open(self.images_path[item])

        convert = transforms.ToTensor()
        img = convert(img)

        sequence_data = img.view(-1)
        mean = sequence_data.mean().item()
        std = sequence_data.std().item()

        sequence_data = add_gaussian_noise(sequence_data, mean, std)

        return img, sequence_data, label

    @staticmethod
    def collate_fn(batch):
        images, sequence, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        sequence = torch.stack(sequence)
        labels = torch.as_tensor(labels)

        return images, sequence, labels
