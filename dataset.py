import glob
import numpy as np
import torch

from torch.utils.data import (Dataset,
                              DataLoader)
from PIL import Image
import torchvision.transforms as transforms
from scipy import io

class OxfordFlowers102Dataset(Dataset):
    def __init__(self,
                 root,
                 transforms_=None,
                 mode='train',
                 attrs=[],
                 missing_ind=False):
        self.transform = transforms.Compose(transforms_)
        ids = np.arange(1, 8189+1)
        indices = np.arange(0, len(ids))
        rnd_state = np.random.RandomState(0)
        rnd_state.shuffle(indices)
        labels = io.loadmat('imagelabels.mat')['labels'].flatten()
        # Shuffle both ids and labels with the same indices.
        labels = labels[indices]
        ids = ids[indices]
        if mode == 'train':
            # Training set is first 90%.
            self.ids = ids[0:int(len(ids)*0.9)]
            self.labels = labels[0:int(len(ids)*0.9)]
        else:
            # Valid set is last 10%.
            self.ids = ids[int(len(ids)*0.9)::]
            self.labels = labels[int(len(ids)*0.9)::]
        self.root = root

    def __getitem__(self, index):
        jpg_name = "image_" + str(self.ids[index]).zfill(5) + ".jpg"
        filepath = "%s/jpg/%s" % (self.root, jpg_name)
        img = self.transform(Image.open(filepath))
        label = torch.LongTensor([self.labels[index]])
        return img, label

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    ds = OxfordFlowers102Dataset(root=".",
                                 transforms_=[transforms.Resize(80),
                                              transforms.RandomCrop(64),
                                              transforms.ToTensor()])
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)

    for xx,yy in loader:
        print(xx.shape)
    
    #xx,yy = iter(loader).next()
    #from torchvision.utils import save_image

    #save_image(xx, "test.png")

    
    #print(xx.shape)
    #print(yy.shape)
