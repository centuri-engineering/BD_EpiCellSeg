#%% Initialize
import napari
import numpy as np
import pandas as pd

from joblib import Parallel, delayed  

from skimage import io
from skimage.transform import resize
from skimage.measure import regionprops, regionprops_table
from skimage.segmentation import watershed, clear_border, find_boundaries
from skimage.morphology import remove_small_objects, label, skeletonize, disk
from skimage.filters import sato, try_all_threshold, threshold_triangle, rank, gaussian, median

from scipy import stats
from scipy.ndimage import binary_fill_holes

#%% Inputs
ROOTPATH = 'D:/CurrentTasks/CENTURI_SummerSchool(2021)/GBE_40x(20s)_67xYW-Frlwt(F2)/'
FILENAME = 'Ctrl_18-05-29_GBE_67xYW(F2)_a04_StackCrop.tif'

RSIZE_FACTOR = 2 # reduce image size by this factor (2)
SATO_SIGMA = 4/RSIZE_FACTOR # sigma size for sato ridge filter (2)
THRESH_COEFF = 1.0 # adjust auto thresholding (the smaller the more sensitive segmentation) (0.5)
MIN_SIZE1 = 4000/RSIZE_FACTOR # minimum size for binary objects (2000 for RSIZE_FACTOR = 2)
MIN_SIZE2 = 100000/RSIZE_FACTOR # minimum size for binary objects (2000 for RSIZE_FACTOR = 2)
BORDER_CUTOFF = 0.75 # remove border cells (from 0 to 1, the smaller the less stringent) 

#%% Open Stack from ROOTPATH+FILENAME

stack = io.imread(ROOTPATH+FILENAME)
nT = stack.shape[0] # Get Stack dimension (t)
nY = stack.shape[1] # Get Stack dimension (y)
nX = stack.shape[2] # Get Stack dimension (x)

#%% Bleach correction

#%% resize stack

def stack_rsize(stack):
    '''Enter function general description + arguments'''
    rsize = resize(stack,(stack.shape[0] // RSIZE_FACTOR, stack.shape[1] // RSIZE_FACTOR), preserve_range=True, anti_aliasing=True) 
    return rsize

output_list = Parallel(n_jobs=35)(
    delayed(stack_rsize)(
        stack[i,:,:]
        )
    for i in range(nT)
    )

rsize = np.stack([arrays for arrays in output_list], axis=0)
nY_rsize = rsize.shape[1] # Get Stack dimension (y)
nX_rsize = rsize.shape[2] # Get Stack dimension (x)

#%% ridge filter (sato)

def stack_ridge(rsize):
    '''Enter function general description + arguments'''
    ridge = sato(rsize,sigmas=SATO_SIGMA,mode='reflect',black_ridges=False) 
    return ridge    
    
output_list = Parallel(n_jobs=35)(
    delayed(stack_ridge)(
        rsize[i,:,:]
        )
    for i in range(nT)
    )

ridge = np.stack([arrays for arrays in output_list], axis=0)

#%% thresholding

# # Test thresholding methods
# import matplotlib.pyplot as plt
# thresh_test = try_all_threshold(ridge[0,:,:], verbose=True)

thresh = threshold_triangle(ridge) # Find thresh
mask = ridge > thresh*THRESH_COEFF # Apply thresh
mask = remove_small_objects(mask, min_size=MIN_SIZE1) # Remove small objects

#%% watershed

def stack_watershed(mask, ridge):
    '''Enter function general description + arguments'''
    labels = label(np.invert(mask), connectivity=1)
    labels = watershed(ridge, labels, watershed_line=False)
    labels = clear_border(labels)
    wat = find_boundaries(labels)
    wat = skeletonize(wat)
    return labels, wat 

output_list = Parallel(n_jobs=35)(
    delayed(stack_watershed)(
        mask[i,:,:],
        ridge[i,:,:]
        )
    for i in range(nT)
    )

labels = np.stack([arrays[0] for arrays in output_list], axis=0)
wat = np.stack([arrays[1] for arrays in output_list], axis=0)
    
#%% clean watershed

# create border mask
border_mask = labels > 0
border_mask = np.mean(border_mask, axis=0)
border_mask = gaussian(border_mask,10)
border_mask = border_mask >= BORDER_CUTOFF
border_wat = wat & border_mask

def stack_clean_watershed(border_wat):
    '''Enter function general description + arguments'''   
    # clean label with border_mask
    labels_clean = label(np.invert(border_wat), connectivity=1)
    labels_clean = watershed(border_wat, labels_clean, watershed_line=False)
    labels_clean = clear_border(labels_clean)
    
    # remove isolated cell
    temp_mask = labels_clean > 0
    temp_mask = remove_small_objects(temp_mask, min_size=MIN_SIZE2)
    labels_clean[temp_mask==0] = 0
    
    # extract cleaned wat
    wat_clean = find_boundaries(labels_clean)
    wat_clean = skeletonize(wat_clean)
    return labels_clean, wat_clean

output_list = Parallel(n_jobs=35)(
    delayed(stack_clean_watershed)(
        border_wat[i,:,:]
        )
    for i in range(nT)
    )

labels = np.stack([arrays[0] for arrays in output_list], axis=0)
wat = np.stack([arrays[1] for arrays in output_list], axis=0)

#%% track & filter cells

small_cell_mask = np.full([nT,nY_rsize,nX_rsize], False)
for i in range(nT):
    temp_im = labels[i,:,:].copy()
    for temp_props in regionprops(temp_im, temp_im): 
        cellID = temp_props.label
        y, x = np.nonzero(temp_im == cellID)
        temp_area = temp_props.area
        if temp_area < 45:
            temp_im[y,x] = True
        else:
            temp_im[y,x] = False
        small_cell_mask[i,:,:] = temp_im

labels_track = labels.copy()
for i in range(1, nT):
    temp_im = labels_track[i,:,:]
    temp_im_prev = labels_track[i-1,:,:]
    for temp_props in regionprops(temp_im, temp_im): 
        cellID = temp_props.label
        y, x = np.nonzero(temp_im == cellID)
        val_prev = temp_im_prev[y,x]
        mod_prev = stats.mode(val_prev)[0]
        if mod_prev > 0:
            temp_im[y,x] = mod_prev
        elif mod_prev == 0:
            temp_im[y,x] = labels_track.max()+1
    labels_track[i,:,:] = temp_im   


#%%

# im = wat_clean[0,:,:]
# def pixconn(im, conn=2):
#     '''Enter function general description + arguments'''
#     conn1_selem = np.array([[0, 1, 0],
#                             [1, 0, 1],
#                             [0, 1, 0]])
#     conn2_selem = np.array([[1, 0, 1],
#                             [0, 0, 0],
#                             [1, 0, 1]])
    
#     conn1 = rank.sum(im.astype('uint8'),conn1_selem)*im
    
#     if conn == 1:
#         im_conn = conn1
#     elif conn == 2:
#         conn2 = rank.sum(im.astype('uint8'),conn2_selem)*im
#         im_conn = conn1 + conn2
#     return im_conn
    
# im_conn = pixconn(im)

#%%

# with napari.gui_qt():
#     viewer = napari.view_image(rsize)
#     viewer = napari.view_image(ridge)
#     viewer = napari.view_image(mask)
    
#%% Save images

io.imsave(ROOTPATH+FILENAME[0:-4]+'_rsize.tif', rsize.astype('uint16'), check_contrast=True)
io.imsave(ROOTPATH+FILENAME[0:-4]+'_ridge.tif', ridge.astype('float32'), check_contrast=True)
io.imsave(ROOTPATH+FILENAME[0:-4]+'_mask.tif', mask.astype('uint8')*255, check_contrast=True)
io.imsave(ROOTPATH+FILENAME[0:-4]+'_labels.tif', labels.astype('uint16'), check_contrast=True)
io.imsave(ROOTPATH+FILENAME[0:-4]+'_wat.tif', wat.astype('uint8')*255, check_contrast=True)
io.imsave(ROOTPATH+FILENAME[0:-4]+'_labels_track.tif', labels_track.astype('uint16'), check_contrast=True)
io.imsave(ROOTPATH+FILENAME[0:-4]+'_small_cell_mask.tif', small_cell_mask.astype('uint8')*255, check_contrast=True)