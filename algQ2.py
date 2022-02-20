import os 
import json 
import copy 
from time import time 
from glob import glob 
from pprint import pprint 

import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

from scipy.linalg import logm, expm
from scipy import optimize, ndimage, interpolate 
from skimage import transform 

DIR_INPUT = '../../input/Q2' 
DIR_OUTPUT = '../../output/Q2'
DIR_EXTRA = os.path.join(DIR_OUTPUT, 'extra')
DIR_ANSWER = os.path.join(DIR_OUTPUT, 'answer') 

for dir_path in [DIR_INPUT, DIR_OUTPUT, DIR_EXTRA]:
    if not os.path.exists(dir_path): os.makedirs(dir_path)  

class Manager:
    def __init__(self, time, size=None, num_laps=20, txt='forloop'):
        self.size = size 
        self.num_laps = 20  
        self.T0 = time
        self.t_old = time
        self.lap = 1
        self.txt = txt 
        print(f'***{txt} starts***')
    
    def step(self, i, time):
        if (i+1) % int(round(self.size/self.num_laps))==0:
            message = str() 
            message += f'{self.lap*int(100/self.num_laps)} % ends: \t'
            message += f'{time-self.t_old:.2f} sec: \t'
            message += f'{time-self.T0:.2f} sec'
            self.lap +=  1
            self.t_old = time
            print(message) 
    
    def end(self,time):
        print(f'***{self.txt} ends*** {time-self.T0:.2f} sec') 

def title(txt='title', length=100, marker='*'):
    print() 
    print(marker*length) 
    print(f'\t {txt.upper()}') 
    print(marker*length) 
    print() 

def subtitle(txt='subtitle', length=100, marker='-'):
    print() 
    print(marker*length) 
    print(f'\t {txt}') 
    print(marker*length) 
    print() 

def subsubtitle(txt='sample', length=35, marker='+'):
    print() 
    print(marker*length) 
    print(f'\t {txt}') 
    print(marker*length) 
    print()

def nameof(var):
    for k,v in globals().items():
        if id(v) == id(var): name_str=k
    return name_str

def pyprint(v, name=None):
    if name==None: name = nameof(v) 
    L = '' if type(v)!='list' else len(v) 
    print() 
    print(f'{name}{type(v),L} = ')
    pprint(v)  

def npprint(array, name=None):
    if name==None: name = nameof(array) 
    print() 
    print(f'{name}{array.shape, array.dtype} = ')
    print(array)  

def jsread(filename):
    return json.load(open(filename))

def jssave(data, name):
    json.dump(data, open(name, 'w'))
    print(f'the data is saved in {name}') 

def kptList2Array(list_kpts):
    list_kpts_sorted = sorted(list_kpts, key=lambda x: x['id'])
    return np.array(list(v['voxel'] for v in list_kpts_sorted)).T

def kptArray2List(np_kpts): 
    return list(dict({'id': k, 'voxel': v}) for k, v in enumerate(np_kpts.T.tolist()))

def imread(filename, shape=[256,256]):
    height, width = shape 
    img = open(filename, 'rb') 
    img = np.fromfile(img, dtype=np.float64, count=height*width) 
    return img.reshape((height, width))

def impoints(img, pts, r=2, color=(0,0,255), thickness=-1):
    img = imrgb(img) 
    for pt in pts:
        img = cv2.circle(img, (round(pt[0]), round(pt[1])), r, color, thickness)
    return img 

def imtxt(image, txt='txt'):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 250)
    fontScale = 0.4
    color = (255, 255, 255)
    thickness = 1
    image = cv2.putText(image, txt, org, font, 
            fontScale, color, thickness, cv2.LINE_AA)
    return image 

def imshow(img, name="img", save=False):
    if save: imsave(img, name) 
    cv2.imshow(name, img)
    while True: 
        if cv2.waitKey(1)==ord('q'): break
    cv2.destroyAllWindows()

def imsave(img, name='img'):
    cv2.imwrite(os.path.join(DIR_EXTRA, f'{name}.png'), img) 

def imrgb(img):
    if len(img.shape) == 2:
        img = (img - np.min(img)) / (np.max(img) - np.min(img)+1e-10) * 255 
        img = img.astype(np.uint8) 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
    return img 

def imcom(image): # Center of Mass 
    com = ndimage.measurements.center_of_mass(image.T) # given as tuple 
    com = np.array([list(com)]) 
    return com 

def imgrad(image):
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) 
    dI = np.zeros((2,*image.shape))
    dI[0] = cv2.filter2D(image, ddepth=-1, kernel=sobel)
    dI[1] = cv2.filter2D(image, ddepth=-1, kernel=sobel.T) 
    return dI 

def imgrid(yx=[8,8],shape=[256,256]):
    height, width = shape[:2]

    dy = height - height//yx[0]
    dx = width - width//yx[1] 
    y = np.linspace(dy,height-dy,yx[0])
    x = np.linspace(dx,width-dx,yx[1]) 

    xx, yy = np.meshgrid(x, y) 
    grid = np.vstack((xx.flatten(), yy.flatten())) 

    return grid.astype(int)

def imdown(volume,f=2):
    return transform.downscale_local_mean(volume, (f, f))

def imrescale(volume, f=(2,2)):
    return transform.rescale(volume, f, mode='constant', anti_aliasing=False)

def kptList2Array(list_kpts):
    return np.array(list(v['pixel'] for v in list_kpts)).T

def kptArray2List(np_kpts): 
    return list(dict({'id': k, 'pixel': v}) for k, v in enumerate(np_kpts.T.tolist()))

def homog(P): # homogenous 
    ones = np.ones(P.shape[1]).reshape(1,-1)
    return np.vstack((P,ones))

def hinv(A):
    sRinv = np.linalg.pinv(A[:2,:2])
    tinv = -A[:2,2].reshape(2,1)
    sRtinv = np.hstack((sRinv, tinv)) 
    bottom = np.array([[0,0,1]])
    Ainv = np.vstack((sRtinv, bottom))
    return Ainv 

def hrescale(A, f):
    upper = np.hstack((A[:2,:2], f*A[:2,2].reshape(2,1)))
    bottom = np.array([[0,0,1]])
    return np.vstack((upper, bottom)) 

def coeff2Affine(coeff): # len(x) = 6
    t = coeff[:2].reshape(2,1)
    R = coeff[2:].reshape(2,2)
    Rt = np.hstack((R, t))
    bottom = np.array([[0,0,1]])
    A = np.vstack((Rt, bottom))
    return A 

def Affine2coeff(A):
    t = A[:2,2] 
    R = A[:2,:2] 
    return np.hstack((t, R.flatten()))

def AffineModel(coeff, image):
    A = coeff2Affine(coeff) 
    projected_image = ndimage.affine_transform(image, hinv(A)) 
    return projected_image 

def difference(coeff, model, source_volume, target_volume):
    diff = model(coeff, source_volume) - target_volume
    return diff.flatten()

def optAffineModel(x_temp, model, source_image, target_image, f=3):
    # downsample 
    x_temp[:2] /= f # rescale for down sample 

    if f>1:
        source_image_f = imdown(source_image, f) 
        target_image_f = imdown(target_image, f)
    else:
        source_image_f = source_image[:] 
        target_image_f = target_image[:] 

    # optimization wrt downsampled voulmes 
    err = lambda x: difference(x, model, source_image_f, target_image_f)   
    ans = optimize.least_squares(err, x_temp, diff_step=None)
    x_opt_f = ans['x']

    # rescale
    x_temp = copy.deepcopy(x_opt_f)     # opt w.r.t. downsampled volumes 
    x_temp[:2] *= f                     # rescale for volumes of original size 

    return x_temp 

def optBSplineModel(x_temp, model, source_image, target_image,size=40):
    # optimization
    err = lambda x: differenceBS(x, model, source_image, target_image,size)   
    ans = optimize.least_squares(err, x_temp, diff_step=None)
    x_opt = ans['x']
    return x_opt 

def differenceBS(coeff, model, source_volume, target_volume, size):
    diff = model(coeff, source_volume, size) - target_volume
    return diff.flatten()


from skimage.transform import warp_coords
from scipy.ndimage import map_coordinates
from scipy.interpolate import BSpline 
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage.map_coordinates
def shift_up10_left20(xy):
    projected_xy =  xy - np.array([-20, 10])[None, :]
    if 0:
        npprint(xy, 'xy')
        npprint(projected_xy, 'shifted_xy') 
    return projected_xy

def affine(xy, coeffs):
    A = coeff2Affine(coeffs) 
    projected_xy = (A @ homog(xy.T))[:-1].T 
    if 0:
        npprint(A, 'A') 
        npprint(xy, 'xy')
        npprint(homog(xy.T), 'alg.homog(xy.T)')
        npprint((A @ homog(xy.T)), '(A @ alg.homog(xy.T))')
        npprint(projected_xy, 'affined_xy')
    return projected_xy

def bspline(xy, coeff, size=30, k=3, n=16, shape=[256,256]):
    w,h = shape
    Cx = coeff[:size**2]; Cy = coeff[size**2:]

    #xlim = (-40,290); ylim = (-40,290)
    xlim = (0,256); ylim = (0,256)
    tx = np.linspace(xlim[0],xlim[1],size)
    ty = np.linspace(ylim[0],ylim[1],size) 

    zx = Cx.reshape(size,size).T
    zy = Cy.reshape(size,size).T 

    if 0:
        fx = interpolate.interp2d(tx, ty, zx, kind='cubic')
        fy = interpolate.interp2d(tx, ty, zy, kind='cubic')
    
    elif 1:
        fx = interpolate.RectBivariateSpline(tx,ty,zx)
        fy = interpolate.RectBivariateSpline(tx,ty,zy)

    x = np.arange(0,256) 
    y = np.arange(0,256)  

    dx = fx(x,y)
    dy = fy(x,y) 

    dxy = np.vstack((dx.flatten(), dy.flatten())).T 
    projected_xy = xy + dxy 

    npprint(xy, 'xy')
    npprint(dx, 'dx') 
    npprint(dy, 'dy') 
    npprint(projected_xy, 'projected_xy')

    return projected_xy 


def BSplineModel(coeffs, image, size=40): 
    coord_map = lambda xy: bspline(xy, coeffs, size) 
    coords = warp_coords(coord_map, image.shape)
    warped_image = map_coordinates(image, coords)
    return warped_image 

'''
t: ndarray, shape (n+k+1,)
knots

c:ndarray, shape (>=n, …)
spline coefficients

k:int
B-spline degree
'''

def nccMatching(source_points, source_patches, target_patches, width=256):
    width = source_patches.shape[1] 
    patch_radius = int(0.5 * (np.sqrt(source_patches.shape[-1]) - 1)) # (2 * patch_radius + 1)**2 = patches.shape[-1] 

    #source_points = source_points + patch_radius 
    source_ids = source_points[1] * width + source_points[0]

    source_patches = source_patches.reshape(-1,(2*patch_radius+1)**2)[source_ids] 
    target_patches = target_patches.reshape(-1,(2*patch_radius+1)**2) 
    ncc = source_patches @ target_patches.T / source_patches.shape[-1]

    target_ids = ncc.argmax(axis=1) 
    target_points = np.vstack((target_ids%width, target_ids//width)) 

    return target_points

def getPatches(gray_source, gray_target, patch_radius):
    height, width = gray_source.shape 
    
    patches_source = np.zeros((height+2*patch_radius, width+2*patch_radius, (2*patch_radius+1)**2)) 
    patches_target = np.zeros((height+2*patch_radius, width+2*patch_radius, (2*patch_radius+1)**2)) 
    gray_source = impad(gray_source, r=patch_radius) 
    gray_target = impad(gray_target, r=patch_radius)

    i = 0
    for y in range(-patch_radius, patch_radius+1):
        for x in range(-patch_radius, patch_radius+1):
            # Shift the image to get different elements of the patch
            patches_source[:,:,i] = np.roll(np.roll(gray_source, -y, axis=0), -x, axis=1) 
            patches_target[:,:,i] = np.roll(np.roll(gray_target, -y, axis=0), -x, axis=1)
            i+=1
    
    patches_source = patches_source[patch_radius:-patch_radius,patch_radius:-patch_radius,:] 
    patches_target = patches_target[patch_radius:-patch_radius,patch_radius:-patch_radius,:] 

    # Compute mean for each patch
    patches_source_mean = patches_source.mean(-1, keepdims=True)
    patches_target_mean = patches_target.mean(-1, keepdims=True)

    # Subtract mean
    patches_source_min_mean = patches_source - patches_source_mean
    patches_target_min_mean = patches_target - patches_target_mean

    # Compute standard deviation for each patch
    patches_source_var = np.square(patches_source_min_mean).mean(-1, keepdims=True)
    patches_source_dev = np.sqrt(patches_source_var) + 1e-8

    patches_target_var = np.square(patches_target_min_mean).mean(-1, keepdims=True)
    patches_target_dev = np.sqrt(patches_target_var) + 1e-8

    # Normalize the patches
    with np.errstate(divide='ignore', invalid='ignore'):  # to suppress zero divison warnings ?? 
        patches_source = patches_source_min_mean / patches_source_dev
        patches_target = patches_target_min_mean / patches_target_dev

    return patches_source, patches_target


def impad(img, r=1, mode='empty'):
    return np.pad(img, pad_width=r, mode=mode) if r>0 else img 
