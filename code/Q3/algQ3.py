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

#from scipy.linalg import logm, expm
from scipy import optimize, ndimage 
from skimage import transform 

DIR_INPUT = '../../input/Q3' 
DIR_OUTPUT = '../../output/Q3'
DIR_EXTRA = os.path.join(DIR_OUTPUT, 'extra')

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
        if id(v) == id(var): name=k
    return name

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

def imtxt(image, txt='txt', org = (50, 250)):
    font = cv2.FONT_HERSHEY_SIMPLEX
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

def imnorm(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img)+1e-10) * 255

def imrgb(img):
    if len(img.shape) == 2:
        img = (img - np.min(img)) / (np.max(img) - np.min(img)+1e-10) * 255 
        img = img.astype(np.uint8) 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
    return img 

def imcom(image):
    com = ndimage.measurements.center_of_mass(image.T) # given as tuple 
    com = np.array([list(com)]) # given as yx 
    return com 

def imgrad(image):
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) 
    dI = np.zeros((2,*image.shape))
    dI[0] = cv2.filter2D(image, ddepth=-1, kernel=sobel)
    dI[1] = cv2.filter2D(image, ddepth=-1, kernel=sobel.T) 
    return dI 

def vmgrad(volume):
    depth, height, width = volume.shape[:3] 
    dV = np.zeros((2,depth,height,width)) 
    for i in range(depth):
        cross = volume[i] 
        dV[:,i]= imgrad(cross)
    return dV 

def vmread(filename): 
    volume = np.load(filename)['voxel']
    volume = volume.transpose(2,0,1)    #!!!
    return volume

def vmshow(volume, name='volume'):
    for z in range(volume.shape[0]):
        cross = volume[z]
        imshow(imrgb(cross), f'{name}_{z}') 

def vmcat(*volumes):
    volume_list = list(volumes) 
    return np.concatenate(volume_list, axis=2) 

def vmhist(volume1, volume2, n_bins=255, range=(0,255),name=f'voxel_histogram'):
    # range 
    range1 = range if volume1.min() >= 0 and volume1.max() <= 255 else (volume1.min(), volume1.max()) 
    range2 = range if volume2.min() >= 0 and volume2.max() <= 255 else (volume2.min(), volume2.max()) 

    h1,b1 = np.histogram(volume1.flatten(),bins=n_bins,range=range1) 
    h2,b2 = np.histogram(volume2.flatten(),bins=n_bins,range=range2) 

    npprint(h1, 'source hist') 
    npprint(h2, 'target hist') 
    npprint(b2, 'bins')

    print('range1 = ', (volume1.min(), volume1.max())) 
    print('range2 = ', (volume2.min(), volume2.max()))

    plt.plot(h1, label='source (T2)') 
    plt.plot(h2, label='target (PD)') 
    plt.yscale("log")
    plt.title(f'{name}') 
    plt.legend() 
    plt.savefig(os.path.join(DIR_EXTRA, name))
    plt.show() 

def vmgrid(zyx=[8,8,8],shape=[130,256,256]):
    depth, height, width = shape[:3]

    dz = depth - depth//zyx[0]
    dy = height - height//zyx[1]
    dx = width - width//zyx[2] 

    z = np.linspace(dz,depth-dz,zyx[0]) 
    y = np.linspace(dy,height-dy,zyx[1])
    x = np.linspace(dx,width-dx,zyx[2]) 

    xxx, yyy, zzz = np.meshgrid(x, y, z) 
    grid = np.vstack((xxx.flatten(), yyy.flatten(), zzz.flatten())) 

    return grid.astype(int)

def vmnorm(volume, minshift=False):
    if minshift:
        hist, bins = np.histogram(volume, bins=255)
        volume_min = bins[hist.argmax()]
        volume[volume < volume_min] = volume_min 

    return (volume - volume.min())/(volume.max() - volume.min()) * 255

def vmdown(volume,f=2):
    return transform.downscale_local_mean(volume, (f, f, f))

def vmrescale(volume, f=(2,2,2)):
    return transform.rescale(volume, f, mode='constant', anti_aliasing=False)

def vmrgb(volume):
    if len(volume.shape) == 3:
        d,h,w = volume.shape
        volume = volume.reshape(d,h,w,1) + np.zeros((d,h,w,3))
    return volume 

def vmpoints(volume, pts, r=2, color=(0,0,255), thickness=-1): 
    volume = vmnorm(volume) 
    volume = vmrgb(volume)
    volume = volume.astype(np.uint8)
    
    for pt in pts:
        z = round(pt[2]); xy = (round(pt[0]), round(pt[1]))
        if z >= volume.shape[2]: continue 
        imz = volume[z].astype(np.uint8)
        volume[z] = cv2.circle(imz, xy, r, color, thickness)
    return volume 

def homog(P): # homogenous 
    ones = np.ones(P.shape[1]).reshape(1,-1)
    return np.vstack((P,ones))

def hinv(M):
    Ainv = np.linalg.pinv(M[:3,:3])
    binv = -Ainv @ M[:3,3].reshape(3,1) # !!! 
    Abinv = np.hstack((Ainv, binv)) 
    bottom = np.array([[0,0,0,1]])
    Minv = np.vstack((Abinv, bottom))
    return Minv 

def hrescale(A, f):
    upper = np.hstack((A[:3,:3], f*A[:3,3].reshape(3,1)))
    bottom = np.array([[0,0,0,1]])
    return np.vstack((upper, bottom)) 

def points3d(points):
    xs = points[0]
    ys = points[1] 
    zs = points[2] 
    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.plot_wireframe(points)
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z') 
    #plt.axis('equal') 
    plt.show()

def rot3d(theta_x, theta_y, theta_z):
    c = np.cos(theta_x); s = np.sin(theta_x) 
    rotx = np.array([
        [ 1, 0, 0],
        [ 0, c,-s],
        [ 0, s, c]
    ])

    c = np.cos(theta_y); s = np.sin(theta_y) 
    roty = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

    c = np.cos(theta_z); s = np.sin(theta_z) 
    rotz = np.array([
        [ c,-s, 0],
        [ s, c, 0],
        [ 0, 0, 1]
    ])

    theta = rotz @ roty @ rotx 
    return theta

def vmviewz(ax, i, volume, ks = [5,50,100],n=100): # volume:zyx
    _, height, width = volume.shape[:3]
    xx, yy = np.meshgrid(np.arange(0,width), np.arange(0,height)) 
    X = xx + i*(width+10)
    Y = yy
    for k in ks:
        Z = k * np.ones(X.shape)
        cross = vmnorm(volume)[k]
        ax.plot_surface(X,Y,Z, rcount=n, ccount=n, facecolors=imrgb(cross)/255, shade=False)
    return ax 

def vmview(volumes=None, points=None, M=np.eye(4), elev=10, azim=-80, name='view',save=True):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f'elev{elev}_azim{azim}')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z') 
    #ax.text(-50, -50, 130, f"M = \n{M}", color='red')

    if volumes!=None:
        num_volumes = len(volumes) 
        for i in range(num_volumes):
            ax = vmviewz(ax, i, volumes[i], ks=[0,30,60,90,110])

    if points!=None:
        colors = ['red', 'blue', 'green'][:len(points)] 
        i = 0
        for i in range(len(volumes)):
            pts = points[i]
            width = 256
            xs = pts[0] + i * (10+width)
            ys = pts[1] 
            zs = pts[2] 
            ax.scatter(xs, ys, zs, color=colors[i%3])

    if save: plt.savefig(os.path.join(DIR_EXTRA, name))
    plt.show()

def solve(source_points, target_points):
    num_pts = source_points.shape[1] 
    Q = np.zeros((3*num_pts,3*4)) 
    Q[0::3,0:3] = source_points.T
    Q[0::3,3]   = 1
    Q[1::3,4:7] = source_points.T 
    Q[1::3,7]   = 1 
    Q[2::3,8:11]= source_points.T 
    Q[2::3,11]  = 1 
    
    # Q source_points = projct_points
    m = np.linalg.pinv(Q.T @ Q) @ Q.T @ target_points.T.flatten().reshape(-1,1) 
    M = np.vstack((m.reshape(3,4),np.array([[0,0,0,1]])))

    if 0:
        npprint(Q, 'Q')  
        npprint(M, 'M')
        npprint(source_points, 'source_kpts') 
        npprint(target_points.astype(int), 'projct_ktps') 
        target_kpts = (M @ homog(source_points))[:-1] 
        npprint(target_kpts.astype(int),'reprojected_kpts') 

    return M

def difference(x, model, source_volume, target_volume):
    diff = model(x, source_volume) - target_volume
    #print(np.sum(diff.flatten()*2))
    return diff.flatten()

def optAffineModel(x_temp, model, source_volume, target_volume, f=3, method='nlls'):
    # downsample 
    x_temp[:3] /= f # rescale for down sample 

    if f>1:
        source_volume_f = vmdown(source_volume, f) 
        target_volume_f = vmdown(target_volume, f)
    else:
        source_volume_f = source_volume[:] 
        target_volume_f = target_volume[:] 
    
    print() 
    # optimization wrt downsampled voulmes 
    if method=='nlls':
        err = lambda x: difference(x, model, source_volume_f, target_volume_f)   
        ans = optimize.least_squares(err, x_temp, diff_step=None, verbose=2)
    elif method=='min':
        err = lambda x: np.sum(difference(x, model, source_volume_f, target_volume_f).flatten()**2)
        ans = optimize.minimize(err, x_temp)
    x_opt_f = ans['x']                  # optimizaed parameters w.r.t. downsampled volumes 

    # rescale
    x_opt = copy.deepcopy(x_opt_f)     
    x_opt[:3] *= f                      # rescale for volumes of original size 

    return x_opt

def AffineModel3d(coeff, volume):
    '''
    scipy.ndimage.affine_transform
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html
    "matrix - ndarray: The inverse coordinate transformation matrix, mapping output coordinates to input coordinates."
    '''
    M = coeff2Affine3d(coeff) 
    projected_volume = ndimage.affine_transform(volume, hinv(M)) 
    #projected_volume = ndimage.affine_transform(volume, M) 
    return projected_volume 

def coeff2Affine3d(x): # len(x) = 12
    b = x[:3].reshape(3,1)
    A = x[3:].reshape(3,3)
    Ab = np.hstack((A, b))
    bottom = np.array([[0,0,0,1]])
    M = np.vstack((Ab, bottom))
    return M

def Affine2coeff3d(M):
    b = M[:3,3] 
    A = M[:3,:3] 
    return np.hstack((b, A.flatten()))
