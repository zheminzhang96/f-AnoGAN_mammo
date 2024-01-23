import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2

from torchvision import transforms
from torch.utils.data.dataset import Dataset
import skimage as ski
from skimage.util import random_noise

#from mammo_process import *
import dataset.image_preprocessing as image_preprocessing

imgSize = 256

class RSNAbreastCancer_Dataset (Dataset):
    def __init__(self, df_data):
        self.df_view = df_data
        #self.traindir = traindir
        #self.imagenames = imagenames
        #self.labels = labels
        self.transformations = transforms.Compose([
                                     transforms.Resize((imgSize,imgSize)),
                                     transforms.ToTensor()
                                    ])
    def __getitem__(self, i):
        #print("i in __getitem__",i)
        df_cc_groups = self.df_view.groupby(['StudyInstanceUID'])
        df_cc_group_i = df_cc_groups.get_group(self.df_view['StudyInstanceUID'].unique()[i])
        # 
        if len(df_cc_group_i[df_cc_group_i['ImageLaterality']=='R']) == 1:
            img_r = Image.open(df_cc_group_i[df_cc_group_i['ImageLaterality']=='R']['png_path'].iloc[0]) #get and open the png path with laterality R 
            #print(df_cc_group_i[df_cc_group_i['ImageLaterality']=='R']['png_path'].iloc[0])
        else:
            exit()
        if len(df_cc_group_i[df_cc_group_i['ImageLaterality']=='L']) == 1:
            img_l = Image.open(df_cc_group_i[df_cc_group_i['ImageLaterality']=='L']['png_path'].iloc[0]) #get and open the png path with laterality L
            #print(df_cc_group_i[df_cc_group_i['ImageLaterality']=='L']['png_path'].iloc[0])
        else:
            exit()
        img_r = np.array(img_r)
        img_l = np.array(img_l)

        # normalize images
        img_r = (((img_r-np.min(img_r))/(np.max(img_r)-np.min(img_r)))*255).astype(dtype='uint8')
        img_l = (((img_l-np.min(img_l))/(np.max(img_l)-np.min(img_l)))*255).astype(dtype='uint8')

        if df_cc_group_i[df_cc_group_i['ImageLaterality']=='R']['PhotometricInterpretation'].iloc[0]=='MONOCHROME1': #invert white background to black background
            #print("INVERT")
            img_r = np.invert(img_r)
        if df_cc_group_i[df_cc_group_i['ImageLaterality']=='L']['PhotometricInterpretation'].iloc[0]=='MONOCHROME1': #invert white background to black background
            #print("INVERT")
            img_l = np.invert(img_l)

        #print(img.shape)
        if check_laterality(img_r, 'R') == True:
            img_crop_r = image_preprocessing.segment_breast(img_r)     
        else:
            img_crop_r = image_preprocessing.segment_breast(img_l)
            
        if check_laterality(img_l, 'L') == True:
            img_crop_l = image_preprocessing.segment_breast(img_l)
        else:
            img_crop_l = image_preprocessing.segment_breast(img_r)
            
        img_conct = image_preprocessing.stitch_images(img_crop_r, img_crop_l)

        #print("noise type:", df_cc_group_i['noise'].iloc[0])
        if df_cc_group_i['noise'].iloc[0] == 'gaussian':
            img_conct = random_noise(np.array(img_conct), mode='gaussian', var=0.1)
            img_conct = np.array(255*img_conct, dtype='uint8')
        if df_cc_group_i['noise'].iloc[0] == 'salt_pepper':
            img_conct = random_noise(np.array(img_conct), mode='s&p',amount=0.6)
            img_conct = np.array(255*img_conct, dtype='uint8')
        if df_cc_group_i['noise'].iloc[0] == 'distort':
            img_conct = distort_img(img_conct)

        #transformations = transforms.Compose([transforms.ToTensor(), transforms.Resize((imgSize,imgSize), antialias=True)])
        transformations = transforms.Compose([transforms.ToTensor(), 
                                          transforms.Resize((imgSize,imgSize), antialias=True),
                                          transforms.RandomRotation(degrees=(-15,15))])
        resize_img = transformations(img_conct)

        # img_conct = np.expand_dims(img_conct, axis=-1)
        # transformation_v2 = v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype(), v2.Resize((imgSize,imgSize), antialias=True), v2.RandomRotation(degrees=(-15, 15)), 
        #                             v2.RandomAdjustSharpness(sharpness_factor=3)])
        # resize_img = transformation_v2(img_conct)
        
        #print("type of resize_img", type(resize_img))
        #print("dtype of resize_img", resize_img.dtype)
        resize_img = resize_img.to(torch.float32)
        return resize_img, df_cc_group_i['label'].iloc[0]
    
    def __len__(self): 
        #print("LEN", len(self.df_view['StudyInstanceUID'].unique()))
        return len(self.df_view['StudyInstanceUID'].unique())
    
def get_breast_data(laterality):
    rootdir = '/mnt/storage/breast_cancer_kaggle/train_images_png/'
    # metadata of dicom files. corresponding dicom file name is in the column file
    meta_df = pd.read_csv(rootdir + 'metadata.csv', dtype='str')
    # mapping from dicom file to pngs 
    maps_df = pd.read_csv(rootdir + 'mapping.csv', dtype='str')
    df_dcm = pd.merge(meta_df, maps_df, left_on='file', right_on='Original DICOM file location')
    df_dcm['png_path'] =df_dcm[' PNG location '].apply(lambda x: x.strip(" "))

    # merge two dataframe to get view
    df_dcm['patient_image_id'] = df_dcm['PatientID'].str.cat(df_dcm['InstanceNumber'].astype(str), sep='.')
    df_org = pd.read_csv("/mnt/storage/breast_cancer_kaggle/train.csv", dtype='str')
    df_org['patient_image_id'] = df_org['patient_id'].str.cat(df_org['image_id'].astype(str), sep='.')
    df = df_dcm.merge(df_org[['patient_image_id', 'view', 'implant', 'cancer']], on='patient_image_id', how='left')

    # sort and drop duplicate records
    df_sort = df.sort_values(['StudyInstanceUID','ContentTime'], ascending=False)
    df_final = df_sort.drop_duplicates(subset=['StudyInstanceUID','ImageLaterality', 'view']) # also the view (add later)

    # get CC/MLO view data
    df_view = df_final[df_final['view']==laterality]

    # exclude implant is 1/true
    df_view = df_view[df_view['implant'] == '0']

    # Split dataset into train, val, and test
    train_ids, test_ids = train_test_split(df_view['PatientID'].unique(),test_size=0.03,random_state=1996)
    df_view.loc[:,('split')] = None 
    df_view.loc[:, ('noise')] = None
    df_view.loc[:, ('label')] = 0
    df_view.loc[df_view['PatientID'].isin(train_ids),'split'] = 'train'  
    # val_ids, test_ids = train_test_split(rem_ids, test_size=0.1, random_state=1996)
    # df_view.loc[df_view['PatientID'].isin(val_ids), 'split'] = 'val'
    df_view.loc[df_view['PatientID'].isin(test_ids),'split'] = 'test' 

    train_df = df_view[df_view['split']=='train']
    #val_df_cc = df_view[df_view['split']=='val']
    test_df = df_view[df_view['split']=='test']

    # Add noise to dataset
    noise_type = ['gaussian', 'none', 'salt_pepper', 'distort']

    train_noise = np.random.choice(noise_type, len(train_df['PatientID'].unique()), p=[0.5, 0.5, 0, 0])
    train_noise_dict = dict(zip(train_df['PatientID'].unique(), train_noise))
    train_df.loc[:,('noise')] = train_df['PatientID'].map(train_noise_dict)
    # val_noise = np.random.choice(noise_type, len(val_df_cc['PatientID'].unique()), p=[0.5, 0.5, 0, 0])
    # val_noise_dict = dict(zip(val_df_cc['PatientID'].unique(), val_noise))
    # val_df_cc.loc[:,('noise')] = val_df_cc['PatientID'].map(val_noise_dict)

    test_noise = np.random.choice(noise_type, len(test_df['PatientID'].unique()), p=[0.2, 0.3, 0.3, 0.2])
    test_noise_dict = dict(zip(test_df['PatientID'].unique(), test_noise))
    test_df.loc[:,('noise')] = test_df['PatientID'].map(test_noise_dict)

    # modify label for test set
    test_df.loc[test_df.noise == 'salt_pepper', 'label'] = 1
    test_df.loc[test_df.noise == 'distort', 'label'] = 1 

    return train_df, test_df

def check_laterality(x, laterality):
    correct_laterality = True
    if laterality == 'R':
        if np.mean(x[:,0:30]) > np.mean(x[:,-30:-1]):  # x[:,0] is the far left
            laterality = 'L'
            correct_laterality = False

    elif laterality == 'L':
        if np.mean(x[:,0:30]) < np.mean(x[:,-30:-1]):
            laterality = 'R'
            correct_laterality = False

    return correct_laterality

def distort_img(image):
    roll_img = np.array(image)
    A = roll_img.shape[0] / 3.0
    w = 2.0 / roll_img.shape[1]
    shift = lambda x: A * np.sin(2.0*np.pi*x * w)
    for i in range(roll_img.shape[0]):
        roll_img[:,i] = np.roll(roll_img[:,i], int(shift(i)))

    return roll_img

