import torch
import numpy as np
from model.VAE import VAE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import torch.nn.functional as F
from PIL import Image

class dataset(Dataset):
    def __init__(self, img) -> None:
        super().__init__()
        self.data = img
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index])

class model():
    def __init__(self) -> None:
        # img = torch.tensor(img)
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.batch_size = 128
        self.model = VAE(32 * 32 * 3, 500, 50).to(self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load("./model_saved/VAE_2.pth"))
    
    def __call__(self, img:torch.tensor):
        res = None
        img_dataset = dataset(img)
        dataloader = DataLoader(img_dataset, self.batch_size, shuffle = False)
        with torch.no_grad():
            for data in dataloader:
                data = data.view(data.shape[0], -1).to(self.device)
                mean, _ = self.model.encoder(data)
                res = torch.cat([res, mean], dim = 0) if not res is None else mean
        res = res.to("cpu").tolist()
        return res
    
if __name__ == "__main__":
    mod = model()
    data = torch.randn([5, 3, 32, 32]).tolist()
    print(mod(data))