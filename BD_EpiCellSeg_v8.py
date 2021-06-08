#%% Initialize
import napari
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.ndimage
from skimage import io
from skimage.transform import resize
from skimage.filters import sato, threshold_triangle, gaussian
from skimage.morphology import remove_small_objects, label, closing, erosion
from skimage.segmentation import watershed, clear_border
from skimage.measure import regionprops, regionprops_table
from skan import skeleton_to_csgraph

#%% Inputs
RootPath = 'D:/CurrentTasks/tempSeg/WI SqhAX;Endocad-GFP, Sqh-Ch(20-08-14)_3'
Filename = '/MAX_WI SqhAX;Endocad-GFP, Sqh-Ch(20-08-14)_3_Ecad.tif'

# RootPath = 'C:\Datas\CurrentData\WI SqhAX;Endocad-GFP, Sqh-Ch(20-08-14)_3'
# Filename = '/MAX_WI SqhAX;Endocad-GFP, Sqh-Ch(20-08-14)_3_Ecad.tif'

# General options
nTCrop = 100 # amount of time-points to be computed
RSizeFactor = 2 # reduce image size by this factor (for test ml = 2)
sigma = 3 # sigma size for sato ridge filter (for test ml = 3)
threshCoeff = 0.5 # adjust auto thresholding (the smaller the more sensitive segmentation) (for test ml = 0.5)

# Advanced options
BinMinSize1 = 2000 # (for test ml = 2000)
BinMinSize2 = 100000 # (for test ml = 2000)

#%% Open stack

# Open Stack from filename
tempStack = io.imread(RootPath+Filename,img_num=0)
nY = tempStack.shape[0] # Get Stack dimension (x)
nX = tempStack.shape[1] # Get Stack dimension (y)
Stack = np.zeros([nTCrop,nY,nX])
for i in range(nTCrop):
    Stack[i,:,:] = io.imread(RootPath+Filename,img_num=i) 
nYRSize = int(nY/RSizeFactor) # Get RSize dimension (x)
nXRSize = int(nX/RSizeFactor) # Get RSize dimension (x)

#%% Bleach correction (mono exponential fit)

# Define fit equation  
def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b

# Measure mean fluo. int. over time
yFluo = np.zeros([nTCrop])
for i in range(nTCrop):
    yFluo[i] = np.mean(Stack[i,:,:])
    
# Perform fit
x = np.linspace(1,nTCrop,nTCrop) 
p0 = (100, .005, 100) # start with values near those we expect
params, _ = scipy.optimize.curve_fit(monoExp, x, yFluo, p0)
m, t, b = params
fitFluo = monoExp(x, m, t, b)
fitFluo = fitFluo/np.amax(fitFluo)
correctFluo = 1/fitFluo

for i in range(nTCrop):
    Stack[i,:,:] = Stack[i,:,:]*correctFluo[i]
    
del yFluo, x, p0, m, t, b, fitFluo, correctFluo
    
#%% Get the raw watershed lines, out = Watershed1

def EpiSegWat(Input):
    # RSize Stack
    RSize = resize(Input,(Input.shape[0] // RSizeFactor, Input.shape[1] // RSizeFactor), preserve_range=True, anti_aliasing=True)    
    # Apply ridge filter (sato)
    RidgeFiltered = sato(RSize,sigmas=sigma,mode='reflect',black_ridges=False)
    # Thresholding
    thresh = threshold_triangle(RidgeFiltered) # Find thresh
    BinaryMask = RidgeFiltered > thresh*threshCoeff # Apply thresh
    # Remove small objects
    BinaryMask = np.array(BinaryMask, dtype=bool) # Convert to boolean
    BinaryMask = remove_small_objects(BinaryMask, min_size=BinMinSize1) # Remove small objects
    # Watershed1
    Watershed1 = np.invert(BinaryMask) # invert images
    Watershed1 = label(Watershed1,connectivity=1) # Find markers
    Watershed1 = watershed(RidgeFiltered, Watershed1, watershed_line=True) # Run watershed
    Watershed1 = Watershed1 < 1 # Get Watershed1 skeleton
    FilledHoles = scipy.ndimage.binary_fill_holes(Watershed1) # Filled hole Watershed1    
  
    return RSize, RidgeFiltered, BinaryMask, Watershed1, FilledHoles

from joblib import Parallel, delayed  
OutputList = Parallel(n_jobs=35)(delayed(EpiSegWat)(Stack[i,:,:]) for i in range(nTCrop))
SubList1 =[OutputList[0] for OutputList in OutputList]
SubList2 =[OutputList[1] for OutputList in OutputList]
SubList3 =[OutputList[2] for OutputList in OutputList]
SubList4 =[OutputList[3] for OutputList in OutputList]
SubList5 =[OutputList[4] for OutputList in OutputList]

# Concatenate the list of returned arrays
RSize = np.stack(SubList1, axis=0)
RidgeFiltered = np.stack(SubList2, axis=0)
BinaryMask = np.stack(SubList3, axis=0)
Watershed1 = np.stack(SubList4, axis=0)
FilledHoles = np.stack(SubList5, axis=0)

del SubList1, SubList2, SubList3, SubList4, SubList5

#%% Clean the raw watershed lines, out = Watershed2

FilledHoles = FilledHoles.astype('uint8')*255
FilledHolesMean = np.mean(FilledHoles,axis=0)
FilledHolesMean = FilledHolesMean>225

def EpiSegClean(Input1,Input2):
    Input1 = Input1.astype('uint8')*255
    Input2 = Input2.astype('uint8')*255 
    tempWat = Input1
    tempWat[Input2==0] = 0
    tempMask = label(np.invert(tempWat), connectivity=1)
    tempMask[tempMask>=2] = 255
    tempMask[tempMask<2] = 0
    tempMask = closing(tempMask)
    tempMask = scipy.ndimage.binary_fill_holes(tempMask)
    tempMask = remove_small_objects(tempMask, min_size=BinMinSize2)
    tempMask = tempMask.astype('uint8')*255 
    tempOutlines = tempMask - erosion(tempMask)
    Watershed2 = (tempWat - np.invert(tempMask)) + tempOutlines
    Watershed2[Watershed2==1] = 0
    
    return Watershed2

from joblib import Parallel, delayed  
OutputList = Parallel(n_jobs=35)(delayed(EpiSegClean)(Watershed1[i,:,:],FilledHolesMean) for i in range(nTCrop))

# Concatenate the list of returned arrays
Watershed2 = np.stack(OutputList, axis=0)
  
#%% Track bounds, out = BoundsLabels

def EpiSegBounds(Input):
    Input = np.array(Input, dtype=bool) # Convert to boolean
    _, _, Branchpoints = skeleton_to_csgraph(Input) # Count skeleton pixels neighbour
    Branchpoints = Branchpoints > 2 # Get > 2 neighbours
    Input = Input.astype(int) # Convert to integer
    Branchpoints = Branchpoints.astype(int) # Convert to integer
    Bounds = Input - Branchpoints
    Bounds = np.array(Bounds, dtype=bool) # Convert to boolean
    BoundsLabels = np.uint16(label(Bounds, connectivity=2)) # Label bounds
    
    return Branchpoints, Bounds, BoundsLabels

Branchpoints = np.zeros([nTCrop,nYRSize,nXRSize])
Bounds = np.zeros([nTCrop,nYRSize,nXRSize])
BoundsLabels = np.zeros([nTCrop,nYRSize,nXRSize])
for i in range(nTCrop):
    Branchpoints[i,:,:], Bounds[i,:,:], BoundsLabels[i,:,:] = EpiSegBounds(Watershed2[i,:,:])

from joblib import Parallel, delayed  
OutputList = Parallel(n_jobs=35)(delayed(EpiSegBounds)(Watershed2[i,:,:]) for i in range(nTCrop))
SubList1 =[OutputList[0] for OutputList in OutputList]
SubList2 =[OutputList[1] for OutputList in OutputList]
SubList3 =[OutputList[2] for OutputList in OutputList]

# Concatenate the list of returned arrays
Branchpoints = np.stack(SubList1, axis=0)
Bounds = np.stack(SubList2, axis=0)
BoundsLabels = np.stack(SubList3, axis=0)

#%% Get bounds properties, out = BoundsProps 

BoundsProps = pd.DataFrame(index=np.arange(0))
for i in range(nTCrop):
    Unique = np.unique(BoundsLabels[i,:,:] )
    Props = pd.DataFrame(index=np.arange(len(Unique)-1),columns=['timepoint', 'label', 'ctrd_X','ctrd_Y', 'maxLength'])
    tempProps = regionprops(BoundsLabels[i,:,:])
    for j in range(len(Unique)-1):
        Props.at[j,'timepoint'] = i
        Props.at[j,'label'] = tempProps[j].label
        Props.at[j,'ctrd_X'] = tempProps[j].centroid[1]
        Props.at[j,'ctrd_Y'] = tempProps[j].centroid[0]  
        Props.at[j,'maxLength'] = tempProps[j].major_axis_length
    BoundsProps = pd.concat([BoundsProps, Props],ignore_index=True) 
    
#%%

BoundsCtrd = np.zeros([nTCrop,nYRSize,nXRSize])
for i in range(nTCrop):
    for j in range (len(BoundsProps)-1):
        if BoundsProps['timepoint'][j] == i:
            ctrd_X = BoundsProps['ctrd_X'][j].astype(int)
            ctrd_Y = BoundsProps['ctrd_Y'][j].astype(int)
            BoundsCtrd[i,ctrd_Y,ctrd_X] = 255

#%% Open stack in napari

# with napari.gui_qt():
#     viewer = napari.view_image(Watershed2)
    
#%% Save datas

# # Change data type
RSize = RSize.astype('uint16')
RidgeFiltered = RidgeFiltered.astype('uint16')
BinaryMask = BinaryMask.astype('uint8')*255
Watershed2 = Watershed2.astype('uint8')
Branchpoints = Branchpoints.astype('uint8')*255
Bounds = Bounds.astype('uint8')*255
BoundsLabels = BoundsLabels.astype('uint16')
BoundsCtrd = BoundsCtrd.astype('uint16')
# FilledHoles = FilledHoles.astype('uint8')*255
# Labels = Labels.astype('uint16')

# tempMask = tempMask.astype('uint8')*255

# # Saving
io.imsave(RootPath+Filename[0:-4]+'_RSize.tif', RSize, check_contrast=True)
io.imsave(RootPath+Filename[0:-4]+'_RidgeFiltered.tif', RidgeFiltered, check_contrast=True)
io.imsave(RootPath+Filename[0:-4]+'_BinaryMask.tif', BinaryMask, check_contrast=True)
io.imsave(RootPath+Filename[0:-4]+'_Watershed2.tif', Watershed2, check_contrast=True)
io.imsave(RootPath+Filename[0:-4]+'_FilledHoles.tif', FilledHoles, check_contrast=True)
io.imsave(RootPath+Filename[0:-4]+'_Branchpoints.tif', Branchpoints, check_contrast=True)
io.imsave(RootPath+Filename[0:-4]+'_Bounds.tif', Bounds, check_contrast=True)
io.imsave(RootPath+Filename[0:-4]+'_BoundsLabels.tif', BoundsLabels, check_contrast=True)
io.imsave(RootPath+Filename[0:-4]+'_BoundsCtrd.tif', BoundsCtrd, check_contrast=True)
# io.imsave(RootPath+Filename[0:-4]+'_Labels.tif', Labels, check_contrast=True)

# io.imsave(RootPath+Filename[0:-4]+'_tempMask.tif', tempMask, check_contrast=True)

