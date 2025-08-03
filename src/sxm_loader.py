"""
Here are the basics for SXM Images

Extracted from PyQt GUI version

@author: Anggara
"""
import numpy as np
import datetime
import matplotlib.pyplot as plt
import scipy.ndimage as snd
import sys
import pickle
import os
import tempfile


if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

# SXM opener
import nanonispy as nap
import scipy.linalg
# Filtering
from scipy.signal.windows import blackmanharris
# from skimage.filters import unsharp_mask
from scipy.ndimage import gaussian_filter as blur

def process_sxm_data(dat):
    """Process SXM data based on the original readSXM function"""
    
    # CALIB MAY 2021 ONWARDS (REF: MUC1Tn Folder --> 210519_Cu100_056.sxm)
    Xscale = 0.861735094047142000
    XYcrosstalk = 0.000476218259800334
    YXcrosstalk = -0.01765821454416330
    Yscale = 0.867834394953107000
    Zscale = 1.000000000000000000
    Psi = 6.800000000000000000
    
    # Raw Z data TRANSPOSED
    Z = dat.signals['Z']['forward'] * 1e10  # in Angs
    
    if True in np.isnan(Z):
        # Find the indices of non-nan values
        rows = np.any(~np.isnan(Z), axis=1)
        cols = np.any(~np.isnan(Z), axis=0)
        
        # Find the bounding box of non-nan values
        row_start, row_end = np.where(rows)[0][[0, -2]]
        col_start, col_end = np.where(cols)[0][[0, -1]]
        
        # Crop the image
        Z = Z[row_start:row_end+1, col_start:col_end+1]
    
    # Flipud corrects for Y-flip due to scan direction
    if dat.header['scan_dir'] == 'down':
        Z = np.flipud(Z)
    
    # Deplaning - rough
    Ztest = Z - plane2(Z)
    Ztest = Ztest - Ztest.min()
    
    # Deplaning - precise
    lim = 0
    count = 0
    Zcheck = Ztest
    while abs(np.median(Zcheck)) > 0.005:
        upp_lim = lim + 0.2
        low_lim = lim
        Zcheck = Z - plane2(Z, mask=
                          np.logical_and(Ztest<(upp_lim*Ztest.max()), Ztest>(low_lim*Ztest.max()))
                          )
        lim += 0.005
        count += 1
        if count > 100:
            print('Precise deplaning fails - using rough planing')
            Zcheck = Ztest
            break
    
    Z = Zcheck
    
    # Calibrate raw image
    calib = [[Xscale, XYcrosstalk, 0],
             [YXcrosstalk, Yscale, 0],
             [0, 0, 1]]
    Zcorr = scipy.ndimage.affine_transform(Z, np.linalg.inv(calib), mode='constant', cval=0)
    
    # Define Zmask
    Zmask = np.ones(Zcorr.shape)
    Zmask[np.where(Zcorr==0)] = False
    Zmask[np.where(Zcorr!=0)] = True
    Zmask = snd.binary_erosion(Zmask, iterations=5)
    
    Pixelsize = dat.header['scan_range']*1e9/dat.header['scan_pixels']  # gives nm/pixel
    
    if Pixelsize[0]-Pixelsize[1] > 1e-2:
        print(f'WARNING - PIXELS ARE NON-SQUARE: {Pixelsize}')
        
        original_shape = Zcorr.shape
        print(f'OLD IMAGE SHAPE: {original_shape} pixels')
        aspect_ratio = original_shape[1] / original_shape[0]
        
        # Calculate the zoom factors
        if aspect_ratio > 1:
            zoom_factors = (aspect_ratio, 1)
        else:
            zoom_factors = (1, 1/aspect_ratio)
        
        # Resize the image
        Zcorr = snd.zoom(Zcorr, zoom_factors)
        Zmask = snd.zoom(Zmask, zoom_factors)
        print(f'NEW IMAGE SHAPE: {Zcorr.shape} pixels')
        
        # New Pixelsize
        Pixelsize = 1e9*dat.header['scan_range']/Zcorr.shape  # nm/px
        print(f'NEW PIXELSIZE: {Pixelsize}')
    
    return Zcorr, Zmask, Pixelsize

def readSXM(File):
    
    try:
        dat = nap.read.Scan(File)
    except:
        print(f'Cannot open {File}')
        return None
        
    # CALIB MAY 2021 ONWARDS (REF: MUC1Tn Folder --> 210519_Cu100_056.sxm)
    Xscale         =    0.861735094047142000
    XYcrosstalk    =    0.000476218259800334
    YXcrosstalk    =    -0.01765821454416330
    Yscale         =    0.867834394953107000
    Zscale         =    1.000000000000000000
    Psi            =    6.800000000000000000
    
    # Raw Z data TRANSPOSED
    Z = dat.signals['Z']['forward'] * 1e10 # in Angs
    
    if True in np.isnan(Z):
        
        # Find the indices of non-nan values
        rows = np.any(~np.isnan(Z), axis=1)
        cols = np.any(~np.isnan(Z), axis=0)
        
        # Find the bounding box of non-nan values
        row_start, row_end = np.where(rows)[0][[0, -2]]
        col_start, col_end = np.where(cols)[0][[0, -1]]
        
        # Crop the image
        Z = Z[row_start:row_end+1, col_start:col_end+1]
    
    # Flipud corrects for Y-flip due to scan direction
    if dat.header['scan_dir'] == 'down': Z = np.flipud(Z)
    
    # Deplaning - rough
    Ztest = Z - plane2(Z)
    Ztest = Ztest - Ztest.min()
    
    # Deplaning - precise
    lim = 0
    count = 0
    Zcheck = Ztest
    while abs(np.median(Zcheck)) > 0.005:
        upp_lim = lim + 0.2
        low_lim = lim
        Zcheck = Z - plane2(Z,mask=
                          np.logical_and( Ztest<(upp_lim*Ztest.max()) , Ztest>(low_lim*Ztest.max()) )
                          )
        lim += 0.005
        count += 1
        if count>100:
            print('Precise deplaning fails - using rough planing')
            Zcheck = Ztest
            break
    # print(np.median(Zcheck)) # This is the surface level
    Z = Zcheck
    
    # Calibrate raw image
    calib = [ [Xscale, XYcrosstalk, 0],
              [YXcrosstalk, Yscale, 0],
              [0, 0, 1] ]
    Zcorr = scipy.ndimage.affine_transform(Z, np.linalg.inv(calib), mode='constant', cval = 0 )
    
    # Define Zmask
    Zmask = np.ones(Zcorr.shape)
    Zmask[np.where(Zcorr==0)] = False
    Zmask[np.where(Zcorr!=0)] = True
    Zmask = snd.binary_erosion(Zmask,iterations=5)
    
    Pixelsize = dat.header['scan_range']*1e9/dat.header['scan_pixels']  # gives nm/pixel
    
    if Pixelsize[0]-Pixelsize[1] > 1e-2:
        print(f'WARNING - PIXELS ARE NON-SQUARE: {Pixelsize}')
        
        # Assuming your image is stored in a variable called 'image'
        original_shape = Zcorr.shape
        print(f'OLD IMAGE SHAPE: {original_shape} pixels')
        aspect_ratio = original_shape[1] / original_shape[0]
        
        # Calculate the zoom factors
        if aspect_ratio > 1:
            zoom_factors = (aspect_ratio, 1)
        else:
            zoom_factors = (1, 1/aspect_ratio)
        
        # Resize the image
        Zcorr = snd.zoom(Zcorr, zoom_factors)
        Zmask = snd.zoom(Zmask, zoom_factors)
        print(f'NEW IMAGE SHAPE: {Zcorr.shape} pixels')
        
        # New Pixelsize
        Pixelsize = 1e9*dat.header['scan_range']/Zcorr.shape #nm/px
        print(f'NEW PIXELSIZE: {Pixelsize}')
    
    # Return processed data instead of setting class variables
    return {
        'originalimg': Zcorr,
        'img': Zcorr,
        'Zmask': Zmask,
        'Pixelsize': Pixelsize,
        'header': dat.header,
        'raw_data': dat,
        'acq_time':dat.header['acq_time'],
        'bias':dat.header['bias']}

def plane2(image, mask=None):
    """
    Corrects the plane of a 2D numpy array image by subtracting the best-fit plane,
    optionally using a mask to determine which pixels are used for plane fitting.
    """
    height, width = image.shape
    y, x = np.mgrid[:height, :width]
    
    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError("Mask must have the same shape as the input image")
        valid_pixels = mask != 0
        x = x[valid_pixels]
        y = y[valid_pixels]
        z = image[valid_pixels]
    else:
        x = x.flatten()
        y = y.flatten()
        z = image.flatten()
    
    A = np.column_stack((x, y, np.ones_like(x)))
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs
    
    y, x = np.mgrid[:height, :width]
    return a*x + b*y + c

def parse(s, delim=' ', ty='str'):
    """Parse string with delimiter"""
    temp = []
    s = s.lstrip()
    while s.find('  ') > 0:
        s = s.replace('  ', ' ')
    while s:
        if delim in s:
            if ty == 'flo':
                temp.append(float(s[0:s.find(delim)].strip()))
            elif ty == 'int':
                temp.append(int(s[0:s.find(delim)].strip()))
            elif ty == 'str':
                temp.append(s[0:s.find(delim)].strip())
            s = s[s.find(delim)+1:].lstrip()
        else:
            if ty == 'flo':
                temp.append(float(s))
            elif ty == 'int':
                temp.append(int(s))
            elif ty == 'str':
                temp.append(s)
            break
    return temp

def nor(v1):
    """Normalize vector"""
    return v1 / np.linalg.norm(v1)

def nlz(image):
    """Normalize image"""
    image = image - image.min()
    return image/image.max()

# Vector calculation functions
def length(v1):
    return np.linalg.norm(v1)

def dist(v1, v2):
    return length(v2-v1)

def uvec(v1):
    return v1/length(v1)

def angle(v1, v2, allpos=True):
    """Calculate angle between vectors. Positive = Clockwise"""
    a = np.degrees(np.arccos(np.dot(uvec(v1), uvec(v2))))
    if allpos: 
        return (a + 360) % 360
    if not allpos: 
        return a

def rot(v1, angle):
    """Rotate vector by angle (CCW)"""
    angle = np.radians(angle)
    v2 = np.zeros(v1.shape)
    v2[0], v2[1] = v1[0]*np.cos(angle) - v1[1]*np.sin(angle), v1[0]*np.sin(angle) + v1[1]*np.cos(angle)
    return v2

def load_sxm_file(file_path, print=False):
    """Load SXM file directly from file path"""
    try:
        sxm_data = readSXM(file_path)
        if sxm_data is not None:
            if print:
                print(f"File loaded successfully: {file_path}")
            return sxm_data
        else:
            if print:
                print(f"Failed to load file: {file_path}")
            return None
    except Exception as e:
        if print:
            print(f"Error processing file: {str(e)}")
        return None
    
