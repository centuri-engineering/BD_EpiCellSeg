#%% Initialize
import napari
import numpy as np

from joblib import Parallel, delayed  

from skimage import io
from skimage.transform import resize
from skimage.segmentation import watershed, clear_border, find_boundaries
from skimage.morphology import remove_small_objects, label, skeletonize, square
from skimage.filters import sato, try_all_threshold, threshold_triangle, rank

from scipy.ndimage import binary_fill_holes 

#%% Inputs
ROOTPATH = 'D:/CurrentTasks/CENTURI_SummerSchool(2021)/GBE_40x(18s)_ctrl-eve-torso/'
FILENAME = 'torso_Endocad-GFP(13-12-13)_03_StackCrop.tif'

RSIZE_FACTOR = 2 # reduce image size by this factor (2)
SATO_SIGMA = 2 # sigma size for sato ridge filter (2)
THRESH_COEFF = 2.0 # adjust auto thresholding (the smaller the more sensitive segmentation) (0.5)
MIN_SIZE1 = 2000 # minimum size for binary objects (2000 for RSIZE_FACTOR = 2)
MIN_SIZE2 = 50000 # minimum size for binary objects (2000 for RSIZE_FACTOR = 2)

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
mask = np.array(mask, dtype=bool) # Convert to boolean
mask = remove_small_objects(mask, min_size=MIN_SIZE1) # Remove small objects

#%% watershed

def stack_watershed(mask, ridge):
    '''Enter function general description + arguments'''
    labels = label(np.invert(mask), connectivity=1)
    labels = watershed(ridge, labels, watershed_line=False)
    labels = clear_border(labels)
    wat = find_boundaries(labels)
    wat = skeletonize(wat)
    return wat, labels

output_list = Parallel(n_jobs=35)(
    delayed(stack_watershed)(
        mask[i,:,:],
        ridge[i,:,:]
        )
    for i in range(nT)
    )

wat = np.stack([arrays[0] for arrays in output_list], axis=0)
labels = np.stack([arrays[1] for arrays in output_list], axis=0)
    
#%% clean watershed

def stack_clean_watershed(wat):
    '''Enter function general description + arguments'''
    # Remove small isolated objects
    temp_mask = binary_fill_holes(wat)
    temp_mask = remove_small_objects(temp_mask, min_size=MIN_SIZE2)
    wat_clean = wat & temp_mask
    
    # 
    
    return wat_clean

output_list = Parallel(n_jobs=35)(
    delayed(stack_clean_watershed)(
        wat[i,:,:]
        )
    for i in range(nT)
    )

wat_clean = np.stack([arrays for arrays in output_list], axis=0)

# im = wat_clean[0,:,:]
# def pixconn(im):
#     conn4_selem = np.array([[0, 1, 0],
#                             [1, 0, 1],
#                             [0, 1, 0]])
#     conn8_selem = np.array([[1, 0, 1],
#                             [0, 0, 0],
#                             [1, 0, 1]])
#     conn4 = rank.sum(im.astype('uint8'),conn4_selem)*im
#     conn8 = rank.sum(im.astype('uint8'),conn8_selem)*im
#     im_conn = test_conn4 + test_conn8
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
io.imsave(ROOTPATH+FILENAME[0:-4]+'_wat_clean.tif', wat_clean.astype('uint8')*255, check_contrast=True)