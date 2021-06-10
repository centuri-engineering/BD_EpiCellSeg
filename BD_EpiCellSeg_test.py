#%% Inputs
ROOT_PATH = 'C:/Users/Benoit Dehapiot/BD_EpiCellSeg/DataSet/'
RSIZE_FACTOR = 2

#%% Get folder list
import os

dir_list = os.listdir(ROOT_PATH)
for stack_name in dir_list:
    if 'rsize' in stack_name: 
        dir_list.remove(stack_name)
        
#%% Batch resize folder content        
        
from skimage import io  
from skimage.transform import resize
     
for stack_name in dir_list:  
    stack_path = (ROOT_PATH+stack_name)
    stack = io.imread(stack_path)
    nT = stack.shape[0] # get Stack size in time
    nY = stack.shape[1] # get Stack size in y
    nX = stack.shape[2] # get Stack size in x
    stack_rsize = resize(stack,(nT//1, nY//RSIZE_FACTOR, nX//RSIZE_FACTOR), preserve_range=True, anti_aliasing=True)     
    stack_rsize = stack_rsize.astype('uint16')
    stack_path_rsize = (ROOT_PATH+stack_name[0:-4]+'_rsize.tif')
    io.imsave(stack_path_rsize, stack_rsize, check_contrast=False)
    