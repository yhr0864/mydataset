import torch
from PIL import Image
from torch.utils.data import Dataset

class Data(Dataset): 
    def __init__(self, datatxt, transform=None, target_transform=None):
        
        fh = open(datatxt, 'r') 
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1]))
            
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        label = Image.open(label).convert('RGB')
     
        if self.transform is not None:
            img = self.transform(img) 
            label = self.transform(label)
        return img,label
 
    def __len__(self): 
        return len(self.imgs)
