import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import random
from pathlib import Path
from glob import glob
import pandas as pd
from PIL import Image
import itertools
#from features import Augmentations


class DataModule(LightningDataModule):
    def __init__(self, dataset, batch_size:int, augment=False):
        super().__init__()
        self.batch_size = batch_size
        self.augment = augment
        self.Dataset = dataset
        #self.syndata = syndata
        #self.cgpart = cgpart

    def setup(self, stage = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = self.Dataset("data/train", self.augment)
            #elf.train_dataset_syn = self.syndata(roots = ["/work3/s194633/PartData_NonRandom/train","/work3/s194633/PartData2/train",
                                                          # "/work3/s194633/PartData2/val","/work3/s194633/PartData_NonRandom/val",
                                                           #  "/work3/s194633/PartData2/test","/work3/s194633/PartData_NonRandom/test"  ],augment=self.augment)
            #self.val_dataset = self.Dataset("data/val", augment=False)
            #self.val_cg = self.cgpart(root = "/work3/s194633/CGPart_real", augs = False)
            #self.val_dataset_syn = self.syndata(roots = ["/work3/s194633/PartData_NonRandom/val","/work3/s194633/PartData2/val",],augment=False)
            #self.val_dataset = self.syndata(roots = ["/work3/s194633/PartData_NonRandom/val","/work3/s194633/PartData2/val",],augment=False)


        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = self.Dataset("data/test", augment=False)

    def train_dataloader(self):
        # self.dataset_train = torch.utils.data.ConcatDataset(
        #     [self.train_dataset, self.train_dataset_syn]
        # ) 

        self.dataset_train = self.train_dataset_syn

        # weights_train = [
        #     [self.dataset_train.__len__() / self.train_dataset.__len__()]
        #     * self.train_dataset.__len__(),
        #     [self.dataset_train.__len__() / self.train_dataset_syn.__len__()]
        #     * self.train_dataset_syn.__len__(),
        # ]
        # weights_train = list(itertools.chain.from_iterable(weights_train))
        # sampler_train = torch.utils.data.WeightedRandomSampler(
        #     weights=weights_train, num_samples=len(weights_train)
        # )
        return DataLoader(self.dataset_train, batch_size=self.batch_size, drop_last=True, num_workers=32,shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_cg, batch_size=2, drop_last=True,num_workers=32)
    
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, drop_last=True)
    
    # TODO: Not sure if this is how the predict dataloader should be, but let it be for now
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1)

class DeloitteDataSet(Dataset):

    def __init__(self, data_dir:str, augment=False):
        self.data_dir = data_dir
        self.augment = augment
        self.images = os.listdir(data_dir)
        self.images = [image for image in self.images if image.endswith(".npy")]
        self.augs = T.Compose([
                                T.RandomResizedCrop(size=256, scale=(0.8, 1.2),interpolation=InterpolationMode.NEAREST),
                                T.RandomHorizontalFlip(),
                                T.RandomRotation(degrees=15),
                                ])
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        self.jit = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # defining nice variables
        image_name = self.images[idx]
        image_path = os.path.join(self.data_dir, image_name)

        # The actual loading
        img = np.load(image_path, allow_pickle=True)
        # Restructuring into image and mask
        image = img[:,:,:3]
        mask = img[:,:,3]

        # Converting to Tensors 
        image = torch.from_numpy(image).permute(2,0,1)
        image = image.float()/255

        mask = torch.from_numpy(mask)
        mask = mask.float()

        mask=mask/10
        mask[mask==9]=0 # Change rest of car to background
        #merge frame and fender
        mask[mask==7]=4
        #image = T.Resize([256,256])(image)
        #mask = T.Resize([256,256], interpolation=InterpolationMode.NEAREST)(mask)

        # Augmentation
        if self.augment:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            image = self.augs(image)
            #image = self.jit(image)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            mask = self.augs(mask.unsqueeze(0))
        # Normalization
        image = self.normalize(image)
            
        
        return image.squeeze(), mask.squeeze()


def fetch_paths(root):
    root = Path(root)
    paths = glob(str(root / "*/*"))
    paths = [glob(path + "/*.png") for path in paths]
    paths = [sorted(path) for path in paths]
    df = pd.DataFrame(paths)
    # ,df.iloc[0,2].split("/")[-1][:5]
    df.columns = df.iloc[0, 0].split("/")[-1][:5], df.iloc[0, 1].split("/")[-1][:4]#, df.iloc[0, 2].split("/")[-1][:4], df.iloc[0, 3].split("/")[-1][:4]
    return df




class SynData(torch.utils.data.Dataset):
    def __init__(self, roots, augment = False):
        self.name = "CrashCar101"
        self.augment = augment
        self.root = roots
        self.paths = [fetch_paths(root) for root in roots]
        self.paths = pd.concat(self.paths, ignore_index=True)
        self.mapindex  = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:4,12:11,13:12,14:13,15:14,16:15,17:16,18:17,19:18,20:19,21:20,22:21,23:22,24:23,25:24,26:25,27:26,28:27}
        self.mapindex2 = {0:0,1:5,2:3,3:0,4:0,5:3,6:0,7:0,8:8,9:2,10:0,11:2,12:0,13:0,14:1,15:4,16:0,17:0,18:0,19:0,20:4,21:0,22:0,23:0,24:0,25:0,26:6,27:0}
        #self.mapindex2 = {0:0,1:5,2:3,3:9,4:10,5:11,6:12,7:13,8:8,9:2,10:14,11:15,12:16,13:17,14:1,15:7,16:18,17:19,18:20,19:21,20:4,21:22,22:23,23:24,24:25,25:26,26:6,27:27}

        self.augs = T.Compose([
                                T.RandomResizedCrop(size=256, scale=(0.5, 2),interpolation=InterpolationMode.NEAREST),
                                T.RandomHorizontalFlip(),
                                T.RandomRotation(degrees=20),
                                ])
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        self.jit = T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.3)
        self.img_trans = T.Compose([T.ToTensor()])
        self.resize = T.Resize([256,256],interpolation=InterpolationMode.NEAREST)
     

    def map_index(self, index):
        return self.mapindex[index]
    
    def map_index2(self, index):
        return self.mapindex2[index]

    
    def __getitem__(self, index):
        img = self.paths.iloc[index]['Image']
        img = Image.open(img) 
        img = self.img_trans(img)

        seg = self.paths.iloc[index]['part']
        seg = T.functional.pil_to_tensor(Image.open(seg))
        seg = seg.long()
        seg = seg.apply_(self.map_index)
        #seg = seg.apply_(self.map_index2)
        seg = seg.squeeze()
        # Augmentation
        if self.augment:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            img = self.jit(img)
            img = self.augs(img)
            #image = self.jit(image)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            seg = self.augs(seg.unsqueeze(0))
        # Normalization
        img = self.normalize(img)
        seg = seg.float()
        return img.squeeze(), seg.squeeze()


    def __len__(self):
        return len(self.paths)




class CgPart(torch.utils.data.Dataset):
    def __init__(self, root, augs=False):
        root = Path(root)
        self.augment = augs
        self.anno_dir = root / "Annotations_png"/ "car_imagenet_cropped"
        self.im_dir = root / "Images" / "car_imagenet_cropped"
        self.anno_paths = glob(str(self.anno_dir / "*"))
        #print(self.anno_paths)
        self.cars = [path.split("/")[-1].split(".")[0] for path in self.anno_paths]
        self.resize_img = T.Resize([256,256])

        self.augs = T.Compose([ T.ToTensor(),
                                T.RandomResizedCrop(size=256, scale=(0.8, 1.2),interpolation=InterpolationMode.NEAREST),
                                T.RandomHorizontalFlip(),
                                T.RandomRotation(degrees=15),
                                ])
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        self.jit = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.resize = T.Resize([256,256],interpolation=InterpolationMode.NEAREST)
        self.resize_img = T.Resize([256,256],interpolation=InterpolationMode.BILINEAR)
        self.totensor = T.ToTensor()
     

        self.mapindex = {0:0,1:1,2:2,3:27,4:3,5:4,6:5,7:27,8:6,9:7,10:8,11:9,12:27,13:10,14:4,15:11,16:27,17:12,18:13,19:14,20:15,21:16,22:17,23:18,24:19,25:20,26:21,27:22,28:23,29:24,30:25,31:26,255:0}

        self.mapindex2 = {0:0,1:5,2:3,3:9,4:10,5:11,6:12,7:13,8:8,9:2,10:14,11:15,12:16,13:17,14:1,15:7,16:18,17:19,18:20,19:21,20:4,21:22,22:23,23:24,24:25,25:26,26:6,27:27}

    def map_index(self, index):
        return self.mapindex[index]
    def map_index2(self, index):
        return self.mapindex2[index]
    
    def __getitem__(self, index):
        car = self.cars[index]
        img = Image.open(self.im_dir / (car + ".JPEG"))
        mask = Image.open(self.anno_dir / (car + ".png"))
        mask = T.functional.pil_to_tensor(mask)
        mask = mask.apply_(self.map_index)
        mask = mask.apply_(self.map_index2)
        mask = mask.long()
        mask = self.resize(mask)
        img = self.resize(img)
        # Augmentation
        if self.augment:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            img = self.augs(img)
            #image = self.jit(image)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            mask = self.augs(mask.unsqueeze(0))
        else:
            img = self.totensor(img)
        
        # Normalization
        img = self.normalize(img)
        mask = mask.float()

        mask = mask.squeeze()
        return img, mask

    def __len__(self):
        return len(self.cars)
