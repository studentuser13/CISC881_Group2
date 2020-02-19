#!/usr/bin/env python
# coding: utf-8

# # Image Preprocessing for the LNDb test set
# #### By Clinton Lau, Donghao Qiao and Fraser Raney
# ## Libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from tqdm import tqdm
# utils.py from the challenge
import utils
from PIL import Image
from skimage.exposure import equalize_hist
from medpy.filter.smoothing import anisotropic_diffusion

# ## Load the Data

# Change this line to the path to the data folders
data_fld = './Test'

data_folders = os.listdir(data_fld)
filepaths_imgs = []

dir_lst = filter(lambda x: x.split('.')[-1] != 'raw', os.listdir(data_fld))
dir_lst = map(lambda x: os.path.join(data_fld, x) ,list(dir_lst))
filepaths_imgs.append(np.array(list(dir_lst)).reshape(-1,1))

filepaths_imgs = np.vstack(filepaths_imgs).reshape(-1,)


# # Save the dataset files
# ## * the axial slides with annotations 
# 
# ### Store necessary information for postprocessing into LNDb Challenge submission format, i.e., [scan id,  nodule prob, x pos, y pos, z pos]
# * Original image shape
# * Origin
# * Transformation matrix
# * LNDb ID (filename)
# * Z-Axis position of Slice/Label

# ## 3 adjacent grayscale axial slides

if not os.path.isdir('sliceTs'):
    os.mkdir('sliceTs')

tsFiles = []
t_wlds = []
origins = []
id_c = 1
id_n = 0
input_shapes = []

for index in tqdm(range(len(filepaths_imgs))):
    sc, sp, o, t = utils.readMhd(filepaths_imgs[index])

    t_img, t_wld = utils.getImgWorldTransfMats(sp, t)

    z_max = sc.shape[0]
    z_max_idx = z_max - (sc.shape[0] % 3)
    for id_n in range(int(z_max_idx * .6),int(z_max_idx * .9), 3):
        
        origins.append(o)
        input_shapes.append(sc.shape)
        slide = sc[int(id_n)-1:int(id_n)+2,:,:]
        slide = equalize_hist(slide)
        slide = anisotropic_diffusion(slide, voxelspacing=sp, kappa=20, gamma=0.01, niter=100, option=2)

        if sc.shape[1:] != (512, 512):
            zoom_factors = np.array((512, 512)) / np.array(sc.shape[1:])
            slide_a = zoom(slide[0], zoom_factors)
            slide_b = zoom(slide[1], zoom_factors)
            slide_c = zoom(slide[2], zoom_factors)
            slide = np.dstack([slide_a, slide_b, slide_c])
        else:
            slide = np.dstack([slide[0], slide[1], slide[2]])

        filename = 'ts-%s-%s.jpg' % (str(filepaths_imgs[index].split('/')[-1].split('.')[0]).zfill(4), str(id_n).zfill(4))
        print(filename)
        tsFiles.append(filename)
        t_wlds.append(filename[:-4])
        np.save(os.path.join('./TransMTs', filename[:-4]), t_wld)
        plt.figure(dpi=100).set_size_inches(6.79,6.79)
        plt.imshow(slide)
        plt.axis('off')
        plt.savefig(os.path.join('./sliceTs', filename), bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close('all')
    
ppt = pd.DataFrame(tsFiles) 

ppt['orig shp'] = pd.Series(input_shapes)
ppt['origin'] = pd.Series(origins)
ppt['t_wrld'] = pd.Series(t_wlds)

# #  Save the ground truth information for training

ppt.to_csv('PPT.csv')

