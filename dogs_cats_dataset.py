import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob

class DogsCatsDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.file_names = glob.glob(root_dir + "*.jpg")
        self.transform = transform
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        x = plt.imread(file_name)
        y = int('cat.' in file_name) # dog 0 and cat 1
        x = self.transform(x)
        return x, y

    def show(self):
        x, y = self.__getitem__(np.random.randint(self.__len__()))
        c = { 0 : 'dog', 1 : 'cat'}
        x = x.detach().permute(1, 2, 0).numpy()
        plt.imshow(x)
        plt.title(c[y])
        plt.show()

def get_data_loaders():
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((128, 128)),
                    transforms.ToTensor()
                ])
    train_dataset = DogsCatsDataset('./dogs_cats_data/train/train/', transform = transform)
    test_dataset = DogsCatsDataset('./dogs_cats_data/test/test/', transform = transform)
    train_loader = DataLoader(train_dataset, batch_size = 10)
    test_loader = DataLoader(test_dataset, batch_size=10)
    return train_loader, test_loader