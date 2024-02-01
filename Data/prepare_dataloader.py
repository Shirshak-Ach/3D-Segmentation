from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import torch
import glob



class MyDataset(Dataset):
    def __init__(self, t2_location, t1ce_location,flair_location,mask_location):
        self.t2_list = sorted(glob.glob(t2_location))
        self.t1ce_list = sorted(glob.glob(t1ce_location))
        self.flair_list = sorted(glob.glob(flair_location))
        self.mask_list = sorted(glob.glob(mask_location))
        self.scaler = MinMaxScaler()
        
    def __len__(self):
        return len(self.t2_list)

    def __getitem__(self, idx):

        temp_image_t2=nib.load(self.t2_list[idx]).get_fdata()
        temp_image_t2=self.scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

    
        temp_image_t1ce=nib.load(self.t1ce_list[idx]).get_fdata()
        temp_image_t1ce=self.scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    
        temp_image_flair=nib.load(self.flair_list[idx]).get_fdata()
        temp_image_flair=self.scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
            
        temp_mask=nib.load(self.mask_list[idx]).get_fdata()

        #     print(type(temp_mask))
        temp_mask=temp_mask.astype(np.uint8)
        temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
        # print(np.unique(temp_mask))

        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
        
        #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
        #cropping x, y, and z
        temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
        temp_mask = temp_mask[56:184, 56:184, 13:141]
        # print(temp_mask.shape)
        # print(np.unique(temp_mask, return_counts=True))
        labels, unique_content_of_label_count = np.unique(temp_mask, return_counts=True)
        # if [1- (0 i.e background / whole figure)] < 0.01 then tya aru segmentation 0.01 vanda kom xa
        if True: #(1 - (unique_content_of_label_count[0]/unique_content_of_label_count.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
            # print("Aru haru forground ma pani values xa and background i.e 0th label chai 0.99 vanda kom xa")

            temp_mask= torch.nn.functional.one_hot(torch.from_numpy(temp_mask).to(torch.int64), num_classes=4)
            temp_combined_images = torch.from_numpy(temp_combined_images)
            
            # Change the order from [batch_size, depth, width, height, channel] to [batch_size, channel, depth, width, height]
            temp_combined_images = temp_combined_images.transpose(0, 3).transpose(2, 3).transpose(1, 2)
            temp_mask = temp_mask.transpose(0, 3).transpose(2, 3).transpose(1, 2)

             #         print(temp_mask.shape)
    #         print(temp_combined_images.shape)
    #         print('BraTS2020_TrainingData/input_data_3channels/images/image_'+str(index)+'.pt')
    #         print('BraTS2020_TrainingData/input_data_3channels/masks/mask_'+str(index)+'.pt')

            # print("Background nai 0.99 vanda besi xa")
            return temp_combined_images, temp_mask
        
    
def get_from_loader(t2_location, t1ce_location, flair_location, mask_location):
    my_dataset = MyDataset(t2_location,t1ce_location,flair_location,mask_location)

    train_ratio = 0.8
    test_ratio = 1.0 - train_ratio
    # Calculate the number of samples for training and testing
    num_samples = len(my_dataset)
    num_train_samples = int(train_ratio * num_samples)
    num_test_samples = num_samples - num_train_samples

    train_dataset, test_dataset = random_split(my_dataset, [num_train_samples , num_test_samples])
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_dataloader, test_dataloader
