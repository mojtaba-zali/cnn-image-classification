from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class DatasetPreparation:
    '''
    This class performs normalization on the dataset to improve the performance of cnn network
    and lso prepares train and test dataset based on physical location of the images (DATA_PATH)
    '''
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_transformations(self):
        return self.transform

    def generate_datasets(self, data_path, train_size, batch_size):
        full_dataset = datasets.ImageFolder(data_path, transform=self.transform)
        full_dataset_loader = DataLoader(dataset=full_dataset, batch_size=batch_size)

        train_size = int(train_size * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

        train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

        return full_dataset_loader, train_dataset_loader, test_dataset_loader, test_dataset