import glob 
from tqdm import tqdm 
import nibabel as nib 
from sklearn.preprocessing import MinMaxScaler
import torch 
import numpy as np




t2_list = sorted(glob.glob('/home/shirshak/Desktop/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
t1ce_list = sorted(glob.glob('/home/shirshak/Desktop/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
flair_list = sorted(glob.glob('/home/shirshak/Desktop/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
mask_list = sorted(glob.glob('/home/shirshak/Desktop/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.nii'))


def get_training_values():
    X_train = []
    y_train = []
    scaler = MinMaxScaler()

    for index in tqdm(range(len(t2_list[:3]))): #Using t1_list as all lists are of same size
        temp_image_t2=nib.load(t2_list[index]).get_fdata()
        temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
    
        temp_image_t1ce=nib.load(t1ce_list[index]).get_fdata()
        temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    
        temp_image_flair=nib.load(flair_list[index]).get_fdata()
        temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
            
        temp_mask=nib.load(mask_list[index]).get_fdata()
    #     print(type(temp_mask))
        temp_mask=temp_mask.astype(np.uint8)
        temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
    #     print(np.unique(temp_mask))

        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
        
        #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
        #cropping x, y, and z
        temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
        temp_mask = temp_mask[56:184, 56:184, 13:141]
    #     print(temp_mask.shape)
    #     print(np.unique(temp_mask, return_counts=True))
        labels, unique_content_of_label_count = np.unique(temp_mask, return_counts=True)
    #     if [1- (0 i.e background / whole figure)] < 0.01 then tya aru segmentation 0.01 vanda kom xa
        if (1 - (unique_content_of_label_count[0]/unique_content_of_label_count.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
    #         print("Aru haru forground ma pani values xa and background i.e 0th label chai 0.99 vanda kom xa")

            temp_mask= torch.nn.functional.one_hot(torch.from_numpy(temp_mask).to(torch.int64), num_classes=4)
            temp_combined_images = torch.from_numpy(temp_combined_images)
            
    #         Change the order from [batch_size, depth, width, height, channel] to [batch_size, channel, depth, width, height]
            temp_combined_images = temp_combined_images.transpose(0, 3).transpose(2, 3).transpose(1, 2)
            temp_mask = temp_mask.transpose(0, 3).transpose(2, 3).transpose(1, 2)
    #         print(temp_mask.shape)
    #         print(temp_combined_images.shape)
    #         print('BraTS2020_TrainingData/input_data_3channels/images/image_'+str(index)+'.pt')
    #         print('BraTS2020_TrainingData/input_data_3channels/masks/mask_'+str(index)+'.pt')
            X_train.append(temp_combined_images)
            y_train.append(temp_mask)
    #         torch.save(temp_combined_images, '/kaggle/working/BraTS2020_TrainingData/input_data_3channels/images/image_'+str(index)+'.pt')
    #         torch.save(temp_mask, '/kaggle/working/BraTS2020_TrainingData/input_data_3channels/masks/mask_'+str(index)+'.pt')
        else:
    #         print("Background nai 0.99 vanda besi xa")
            continue
    
    return X_train, y_train


def get_testing_values():
    X_test = []
    y_test = []
    scaler = MinMaxScaler()

    for index in tqdm(range(len(t2_list[3:4]))): #Using t1_list as all lists are of same size
        temp_image_t2=nib.load(t2_list[index]).get_fdata()
        temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
    
        temp_image_t1ce=nib.load(t1ce_list[index]).get_fdata()
        temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
    
        temp_image_flair=nib.load(flair_list[index]).get_fdata()
        temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
            
        temp_mask=nib.load(mask_list[index]).get_fdata()
    #     print(type(temp_mask))
        temp_mask=temp_mask.astype(np.uint8)
        temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
    #     print(np.unique(temp_mask))

        temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
        
        #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
        #cropping x, y, and z
        temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
        temp_mask = temp_mask[56:184, 56:184, 13:141]
    #     print(temp_mask.shape)
    #     print(np.unique(temp_mask, return_counts=True))
        labels, unique_content_of_label_count = np.unique(temp_mask, return_counts=True)
    #     if [1- (0 i.e background / whole figure)] < 0.01 then tya aru segmentation 0.01 vanda kom xa
        if (1 - (unique_content_of_label_count[0]/unique_content_of_label_count.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
    #         print("Aru haru forground ma pani values xa and background i.e 0th label chai 0.99 vanda kom xa")

            temp_mask= torch.nn.functional.one_hot(torch.from_numpy(temp_mask).to(torch.int64), num_classes=4)
            temp_combined_images = torch.from_numpy(temp_combined_images)
            
    #         Change the order from [batch_size, depth, width, height, channel] to [batch_size, channel, depth, width, height]
            temp_combined_images = temp_combined_images.transpose(0, 3).transpose(2, 3).transpose(1, 2)
            temp_mask = temp_mask.transpose(0, 3).transpose(2, 3).transpose(1, 2)
    #         print(temp_combined_images.shape)
    #         print(temp_mask.shape)
    #         print('BraTS2020_TrainingData/input_data_3channels/images/image_'+str(index)+'.pt')
    #         print('BraTS2020_TrainingData/input_data_3channels/masks/mask_'+str(index)+'.pt')
            X_test.append(temp_combined_images)
            y_test.append(temp_mask)
    #         torch.save(temp_combined_images, '/kaggle/working/BraTS2020_TrainingData/input_data_3channels/images/image_'+str(index)+'.pt')
    #         torch.save(temp_mask, '/kaggle/working/BraTS2020_TrainingData/input_data_3channels/masks/mask_'+str(index)+'.pt')
        else:
    #         print("Background nai 0.99 vanda besi xa")
            continue

    return X_test, y_test























