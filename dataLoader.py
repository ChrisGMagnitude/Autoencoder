import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import v2

import h5py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MagClassDataset(Dataset):

    def __init__(self, hdf5_file, 
                 augment=True, crop_ranges=[[-1,2],[-3,5],[-10,20]], crop_jitter=[0.25,0.5,2], max_white_noise=0.05,binary_label=True):
        """
        Arguments:
            csv_file (string): Path to the HDF5 file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hdf5_file = hdf5_file
        self.fh = h5py.File(self.hdf5_file, "r")
        self.classes = np.unique(self.labels)
        self.crop_ranges = crop_ranges
        self.crop_jitter = crop_jitter
        self.max_white_noise = max_white_noise
        self.augment = augment
           
        
        
        
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.fh["images"][idx]
        image[np.isnan(image)] = 0
        image = self.clip_and_normalise_data(image)
        image = self.apply_transforms(image)

        sample = [image]
        

        return sample
        
    def clip_and_normalise_data(self,image):
        
        output = []
        
        
        for i,r in enumerate(self.crop_ranges):
            
            # Clip data
            if self.augment:
                jitter = np.random.uniform(low=-self.crop_jitter[i],
                                       high=self.crop_jitter[i],
                                       size=2)
                                       
                low_clip = r[0]+jitter[0]    
                high_clip = r[1]+jitter[1] 

            else:
                low_clip = r[0]   
                high_clip = r[1] 
                
            im = np.clip(image, low_clip, high_clip)
            
            # Normalise data
            im = (im - low_clip)/(high_clip - low_clip)
            
            
            
            output.append(im)
            
        
        return np.array(output)
        
    def apply_transforms(self,image):
        
        image = torch.from_numpy(image)
        
        
        crop_size = int(round(np.sqrt(image.shape[1]**2/2)))
        
        rand = np.random.uniform(low=0,high=self.max_white_noise)
        
        if self.augment:
            transformer = transforms.Compose([
                                            transforms.v2.CenterCrop((crop_size,crop_size)),
                                            transforms.v2.RandomRotation(degrees=(0, 360)),
                                            transforms.v2.RandomHorizontalFlip(),
                                            transforms.v2.GaussianNoise(sigma=rand)
                                            ])
        else:
            transformer = transforms.Compose([
                                            transforms.v2.CenterCrop((crop_size,crop_size)),
                                            ])
            
            
        image = transformer(image)
        
        return image.type(torch.float)
        
        
def make_weights_for_balanced_classes(classes, nclasses):  
    classes = classes.astype(int)
    count = [0] * nclasses       
    for item in classes:          
        count[item] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(classes)                                              
    for idx, val in enumerate(classes):                     
        weight[idx] = weight_per_class[val]                                  
    return weight



                                                        
def get_weighted_data_loader(dataset,epoch_size,batch_size,num_workers=40):                                                        
    # For unbalanced dataset we create a weighted sampler   
    weights = make_weights_for_balanced_classes(dataset.labels, len(dataset.classes))     
                                                            
    weights = torch.DoubleTensor(weights)
    
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, epoch_size)#len(weights))                     
                                                                                    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,                               
                                             sampler = sampler, pin_memory=True,num_workers=num_workers)      
    return data_loader
                                             