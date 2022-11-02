from torchvision import transforms
from torch.utils.data import Dataset

import medmnist
from medmnist import INFO

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def load_data(args):
    
    info = INFO[args.dataset]
    DataClass = getattr(medmnist, info['python_class'])
    trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.5], std=[.5])])

    dataset_train = DataClass(split='train', transform=trans, download=True, as_rgb=True)
    dataset_test = DataClass(split='test', transform=trans, download=True, as_rgb=True)
    dataset_val = DataClass(split='val', transform=trans, download=True, as_rgb=True)
    
    return dataset_train, dataset_test, dataset_val
