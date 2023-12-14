
import pandas as pd
import glob 
from tqdm import tqdm 
import nibabel as nib 


def check_voxels_resolution_and_size():
    t1_list = sorted(glob.glob('/media/shirshak/E076749B767473DE/3DsegmentationBRATSDataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1.nii'))
    t2_list = sorted(glob.glob('/media/shirshak/E076749B767473DE/3DsegmentationBRATSDataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
    t1ce_list = sorted(glob.glob('/media/shirshak/E076749B767473DE/3DsegmentationBRATSDataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
    flair_list = sorted(glob.glob('/media/shirshak/E076749B767473DE/3DsegmentationBRATSDataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
    mask_list = sorted(glob.glob('/media/shirshak/E076749B767473DE/3DsegmentationBRATSDataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.nii'))

    columns = ['t1_shape', 't2_shape', 't1ce_shape', 'flair_shape', 'mask_shape']
    df = pd.DataFrame(columns=columns)

    for index in tqdm(range(len(t2_list[:15]))):
        temp_image_t1=nib.load(t1_list[index]).get_fdata()
    #     print(temp_image_t1.shape)
        voxel_sizes_t1 = nib.load(t1_list[index]).header.get_zooms()
        
        temp_image_t2=nib.load(t2_list[index]).get_fdata()
        voxel_sizes_t2 = nib.load(t2_list[index]).header.get_zooms()
        
        temp_image_t1ce=nib.load(t1ce_list[index]).get_fdata()
        voxel_sizes_t1ce = nib.load(t1ce_list[index]).header.get_zooms()
        
        temp_image_flair=nib.load(flair_list[index]).get_fdata()
        voxel_sizes_flair = nib.load(flair_list[index]).header.get_zooms()
        
        temp_mask=nib.load(mask_list[index]).get_fdata()
        voxel_sizes_mask = nib.load(mask_list[index]).header.get_zooms()
        

    #     print(temp_image_t2.shape)
        df.loc[index+1, 't1_shape'] = temp_image_t1[index].shape, voxel_sizes_t1
        df.loc[index+1, 't2_shape'] = temp_image_t2[index].shape, voxel_sizes_t2
        df.loc[index+1, 't1ce_shape'] = temp_image_t1ce[index].shape, voxel_sizes_t1ce
        df.loc[index+1, 'flair_shape'] = temp_image_flair[index].shape, voxel_sizes_flair
        df.loc[index+1, 'mask_shape'] = temp_mask[index].shape, voxel_sizes_mask

    return df

if __name__ == '__main__':
    df = check_voxels_resolution_and_size()
    # print(df)
    df.to_csv('utils/sizes_and_voxels.csv')