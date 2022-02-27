import os 
import copy 
import time 
from datetime import datetime 
from pprint import pprint

import cv2 
import numpy as np 
import torch 
import json 

from matplotlib import pyplot as plt 
import scipy
from skimage import transform
from skimage.feature import SIFT, match_descriptors, plot_matches 

def main():

    args = {
        'data_dir': '../../input/Q2',
        'ans_dir': '../../output/Q2'
    }  

    if 0:
        show_loaded_data(args)

    # keypoint matching 
    patch_radius = 9 
    runNccMatching(args, patch_radius, flag='kpts') # nccによるtarget keypointsの推定
    runNccMatching(args, patch_radius, flag='grid') # nccによる任意の格子点の対応関係の推定
    if 0: showNccMatching(args, patch_radius=9, flag='grid') # nccによる推定結果の表示
    polish3(args,patch_radius) # nccによる結果を、Affine変換モデルなどにより洗練
   
def polish3(args, patch_radius=9): 
    title(f'polish gradient: patch_radius = {patch_radius}') 

    source_list = jsread(os.path.join(args['data_dir'], 'keypoints_source.json')) 
    target_list = jsread(os.path.join(args['ans_dir'], 'ncc_target_list', f'pr{patch_radius}.json'))
    grids_list = jsread(os.path.join(args['ans_dir'], 'ncc_grids_list', f'pr{patch_radius}.json'))

    thr1 = 9; thr2 = 9; thr3 = 6; n = 3 # best 
    polish_list = copy.deepcopy(target_list) 
    polish_dir = os.path.join(args['ans_dir'], 'polish3')
    if not os.path.exists(polish_dir): os.makedirs(polish_dir) 
    
    numDataSets = len(source_list) 
    print('***computation starts***'); t0 = time.time(); lap = 1; num_laps = 20
    for i in range(numDataSets):
        if (i+1) % int(numDataSets/num_laps)==0:
            print(f'{lap*int(100/num_laps)} % ends: {time.time()-t0:.2f} sec'); lap +=  1
        
        # ====================================================================================
        #   input 
        # ====================================================================================
        # info 
        source = source_list[i] 
        filename = source['filename'] 

        # kpts 
        source_kpts = getNumpyKpts(source_list[i])
        target_kpts = getNumpyKpts(target_list[i])

        # grid 
        source_grid = np.array(grids_list[i]['source_grid']).T
        target_grid = np.array(grids_list[i]['target_grid']).T 

        # Affine fitting using all the grid points 
        A1 = DirectLinearTransformation(source_grid, target_grid) 
        projected_grid = (A1 @ homogenous(source_grid))[:-1]
        err1 = np.sqrt(np.sum((projected_grid - target_grid)**2, axis=0)) 
        ids1 = (err1 < thr1) 

        # Affine fitting only using inliers 
        A2 = DirectLinearTransformation(source_grid[:,ids1], target_grid[:,ids1])
        projected_grid = (A2 @ homogenous(source_grid))[:-1] 
        err2 = np.sqrt(np.sum((projected_grid - target_grid)**2, axis=0)) 
        ids2 = (err2 < thr2) 

        # polish the Affine model estimation by gradient method 
        A3 = GradientMethod(source_grid[:,ids2], target_grid[:,ids2], A2) 
        projected_kpts = (A3 @ homogenous(source_kpts))[:-1]
        projected_grid = (A3 @ homogenous(source_grid))[:-1] 
        #projected_kpts = (A2 @ homogenous(source_kpts))[:-1]
        #projected_grid = (A2 @ homogenous(source_grid))[:-1] 

        #---------------------------------------------------------------------------------------
        # this part deals with exeptional samples (those with different intensity ranges) 
        #print(target_grid.shape[1])
        if target_grid.shape[1] < 100:
            source_image = imread(os.path.join(args['data_dir'], 'images_source', filename)) 
            target_image = imread(os.path.join(args['data_dir'], 'images_target', filename))

            source_patches, target_patches = getPatches(source_image, target_image, patch_radius) 
            source_grid = getSourceGrid(source_image, grid_n=40, area_threshold=20) 
            target_grid = nccMatching(source_grid, source_patches, target_patches)

            A1 = DirectLinearTransformation(source_grid, target_grid) 
            projected_grid = (A1 @ homogenous(source_grid))[:-1]
            err1 = np.sqrt(np.sum((projected_grid - target_grid)**2, axis=0)) 
            ids1 = (err1 < thr1) 

            A2 = DirectLinearTransformation(source_grid[:,ids1], target_grid[:,ids1])
            projected_grid = (A2 @ homogenous(source_grid))[:-1] 
            err2 = np.sqrt(np.sum((projected_grid - target_grid)**2, axis=0)) 
            ids2 = (err2 < thr2) 

            A3 = GradientMethod(source_grid[:,ids2], target_grid[:,ids2], A2) 
            projected_kpts = (A3 @ homogenous(source_kpts))[:-1]
            projected_grid = (A3 @ homogenous(source_grid))[:-1] 
            #projected_kpts = (A2 @ homogenous(source_kpts))[:-1]
            #projected_grid = (A2 @ homogenous(source_grid))[:-1] 
        #---------------------------------------------------------------------------------------
        vec = target_grid - projected_grid 
        u = np.zeros_like(projected_kpts) 
        for k in range(u.shape[1]):
            # determine a threshold: 
            dist = np.sqrt(np.sum((target_grid - projected_kpts[:,k].reshape(2,1))**2, axis=0))
            thr = sorted(dist)[n] #if n < len(sorted(dist)) else sorted(dist)[-1] 
            ids = (dist <= thr)
            u[:,k] = vec[:, ids].mean(axis=1)
        
        transformed_kpts = projected_kpts + u 
        err3 = np.sqrt(np.sum((transformed_kpts - target_kpts)**2, axis=0)) 
        ids3 = (err3 > thr3)
        #---------------------------------------------------------------------------------------
        err3 = np.sqrt(np.sum((projected_kpts - target_kpts)**2, axis=0))
        ids3 = (err3 > thr3) 

        polish_kpts = copy.deepcopy(target_kpts) 
        polish_kpts[:,ids3] = projected_kpts[:,ids3]
        #polish_kpts[:,ids3] = transformed_kpts[:,ids3]
        polish_list[i]['keypoints'] = toKptList(polish_kpts)

        if 0:
            source_image = imread(os.path.join(args['data_dir'], 'images_source', filename)) 
            target_image = imread(os.path.join(args['data_dir'], 'images_target', filename))

            # kpts image 
            image00 = impoints(source_image, source_kpts.T) 
            image01 = impoints(target_image, polish_kpts.T) 
            image02 = impoints(target_image, target_kpts.T, color=(0,0,255))
            image02 = impoints(image02, transformed_kpts.T, color=(0,255,0))
            image0 = np.hstack((image00, image01, image02))

            # grid image 
            projected_grid = (A2 @ homogenous(source_grid))[:-1]
            err = np.sqrt(np.sum((projected_grid - target_grid)**2,axis=0))
            polish_grid = copy.deepcopy(target_grid) 
            polish_grid[:,err > thr2] = projected_grid[:,err > thr2] 

            image10 = impoints(source_image, source_grid.T) 
            image11 = impoints(target_image, polish_grid.T) 
            image12 = impoints(target_image, target_grid.T, color=(0,0,255))
            image12 = impoints(image12, projected_grid.T, color=(0,255,0)) 
            image1 = np.hstack((image10, image11, image12)) 

            # final image 
            image = np.vstack((image0, image1))
            imshow(image, f'{i}_filename') 

    print(f'***computation ends: {time.time() - t0}')

    jssave(polish_list, os.path.join(polish_dir, f'pr{patch_radius}_all{thr1}_part{thr2}_grad{thr3}_n{n}.json')) 

def polishGradient(args, patch_radius=9): 
    title(f'polish gradient: patch_radius = {patch_radius}') 

    source_list = jsread(os.path.join(args['data_dir'], 'keypoints_source.json')) 
    target_list = jsread(os.path.join(args['ans_dir'], 'ncc_target_list', f'pr{patch_radius}.json'))
    #target_list = jsread(os.path.join(args['ans_dir'], f'test_pr{patch_radius}_202112041546.json'))
    grids_list = jsread(os.path.join(args['ans_dir'], 'ncc_grids_list', f'pr{patch_radius}.json'))
    #grids_list = npload(os.path.join(args['ans_dir'], 'ncc_grids_list', f'pr{patch_radius}.npz'))

    thr1 = 9; thr2 = 9; thr3 = 6 # best 
    polish_list = copy.deepcopy(target_list) 
    polish_dir = os.path.join(args['ans_dir'], 'polish')
    if not os.path.exists(polish_dir): os.makedirs(polish_dir) 
    
    numDataSets = len(source_list) 
    print('***computation starts***'); t0 = time.time(); lap = 1; num_laps = 20
    for i in range(numDataSets):
        if (i+1) % int(numDataSets/num_laps)==0:
            print(f'{lap*int(100/num_laps)} % ends: {time.time()-t0:.2f} sec'); lap +=  1
        source = source_list[i] 
        filename = source['filename'] 

        source_kpts = getNumpyKpts(source_list[i])
        target_kpts = getNumpyKpts(target_list[i])

        source_grid = np.array(grids_list[i]['source_grid']).T
        target_grid = np.array(grids_list[i]['target_grid']).T 

        A1 = DirectLinearTransformation(source_grid, target_grid) 
        projected_grid = (A1 @ homogenous(source_grid))[:-1]
        err1 = np.sqrt(np.sum((projected_grid - target_grid)**2, axis=0)) 
        ids1 = (err1 < thr1) 

        A2 = DirectLinearTransformation(source_grid[:,ids1], target_grid[:,ids1])
        projected_grid = (A2 @ homogenous(source_grid))[:-1] 
        err2 = np.sqrt(np.sum((projected_grid - target_grid)**2, axis=0)) 
        ids2 = (err2 < thr2) 

        A3 = GradientMethod(source_grid[:,ids2], target_grid[:,ids2], A2) 
        projected_kpts = (A3 @ homogenous(source_kpts))[:-1]
        err3 = np.sqrt(np.sum((projected_kpts - target_kpts)**2, axis=0)) 
        ids3 = (err3 > thr3)

        polish_kpts = copy.deepcopy(target_kpts) 
        polish_kpts[:,ids3] = projected_kpts[:,ids3]
        polish_list[i]['keypoints'] = toKptList(polish_kpts)

        if 0:
            source_image = imread(os.path.join(args['data_dir'], 'images_source', filename)) 
            target_image = imread(os.path.join(args['data_dir'], 'images_target', filename))

            # kpts image 
            image00 = impoints(source_image, source_kpts.T) 
            image01 = impoints(target_image, polish_kpts.T) 
            image02 = impoints(target_image, target_kpts.T, color=(0,0,255))
            image02 = impoints(image02, projected_kpts.T, color=(0,255,0))
            image0 = np.hstack((image00, image01, image02))

            # grid image 
            projected_grid = (A2 @ homogenous(source_grid))[:-1]
            err = np.sqrt(np.sum((projected_grid - target_grid)**2,axis=0))
            polish_grid = copy.deepcopy(target_grid) 
            polish_grid[:,err > thr2] = projected_grid[:,err > thr2] 

            image10 = impoints(source_image, source_grid.T) 
            image11 = impoints(target_image, polish_grid.T) 
            image12 = impoints(target_image, target_grid.T, color=(0,0,255))
            image12 = impoints(image12, projected_grid.T, color=(0,255,0)) 
            image1 = np.hstack((image10, image11, image12)) 

            # final image 
            image = np.vstack((image0, image1))
            imshow(image, f'{i}_filename') 

    print(f'***computation ends: {time.time() - t0}')

    jssave(polish_list, os.path.join(polish_dir, f'pr{patch_radius}_all{thr1}_part{thr2}_grad{thr3}.json')) 

def polish2(args, patch_radius=9): 
    title(f'polish 2: patch_radius = {patch_radius}') 

    source_list = jsread(os.path.join(args['data_dir'], 'keypoints_source.json')) 
    #target_list = jsread(os.path.join(args['ans_dir'], 'ncc_target_list', f'pr{patch_radius}.json'))
    target_list = jsread(os.path.join(args['ans_dir'], f'test_pr{patch_radius}_202112041546.json'))
    grids_list = jsread(os.path.join(args['ans_dir'], 'ncc_grids_list', f'pr{patch_radius}.json'))
    #grids_list = npload(os.path.join(args['ans_dir'], 'ncc_grids_list', f'pr{patch_radius}.npz'))

    thr1 = 9; thr2 = 6; r = 3; #width = 256 
    polish_list = copy.deepcopy(target_list) 
    polish_dir = os.path.join(args['ans_dir'], 'polish2')
    if not os.path.exists(polish_dir): os.makedirs(polish_dir) 
    
    numDataSets = len(source_list) 
    print('***computation starts***'); t0 = time.time(); lap = 1; num_laps = 20
    for i in range(numDataSets):
        if (i+1) % int(numDataSets/num_laps)==0:
            print(f'{lap*int(100/num_laps)} % ends: {time.time()-t0:.2f} sec'); lap +=  1
    
        source = source_list[i] 
        filename = source['filename'] 
        source_image = imread(os.path.join(args['data_dir'], 'images_source', filename)) 
        target_image = imread(os.path.join(args['data_dir'], 'images_target', filename))

        source_kpts = getNumpyKpts(source_list[i])
        target_kpts = getNumpyKpts(target_list[i])

        source_grid = np.array(grids_list[i]['source_grid']).T
        target_grid = np.array(grids_list[i]['target_grid']).T 

        A1 = DirectLinearTransformation(source_grid, target_grid) 
        projected_grid = (A1 @ homogenous(source_grid))[:-1]
        err1 = np.sqrt(np.sum((projected_grid - target_grid)**2, axis=0)) 
        ids1 = (err1 < thr1) 

        A2 = DirectLinearTransformation(source_grid[:,ids1], target_grid[:,ids1])
        projected_kpts = (A2 @ homogenous(source_kpts))[:-1]
        err2 = np.sqrt(np.sum((projected_kpts - target_kpts)**2, axis=0)) 
        ids2 = (err2 > thr2)

        polish_kpts = copy.deepcopy(target_kpts) 
        polish_kpts[:,ids2] = projected_kpts[:,ids2]
        polish_list[i]['keypoints'] = toKptList(polish_kpts)

        # ------------------------------------------------------------------------------------------------
        # get patches 
        numKpts = source_kpts.shape[1] 
        source_patches, target_patches = getPatches(source_image, target_image, patch_radius)
        src_patches = source_patches[source_kpts[1],source_kpts[0]].reshape(numKpts, (2*patch_radius+1)**2) 
        tgt_patches = target_patches[target_kpts[1],target_kpts[0]].reshape(numKpts, (2*patch_radius+1)**2)
        projected_xy = projected_kpts.round().astype(np.int64).T
        #print(projected_xy)

        #print('-'*100)
        for j in range(numKpts):
            # compute ncc between source and target keypoint j 
            # get patches surrounding keypoint j  
            src_patch_j = src_patches[j] 
            tgt_patch_j = tgt_patches[j] 
            ncc_base_j = np.sum(src_patch_j * tgt_patch_j) / (2*patch_radius+1)**2

            # compute ncc between source keypoint j and pixels around the projected keypoint j
            # get a patch around the projected keypoint j 
            # reshape the patch so that we can compute ncc of the patch w.r.t. source kpt j 
            Oj = np.maximum(0, projected_xy[j]-r) 
            Tj = np.minimum(projected_xy[j]+r+1, 256) 
            width_j = Tj[0] - Oj[0]
            prj_patch_j = target_patches[Oj[1]:Tj[1], Oj[0]:Tj[0]] # need padding ?
            prj_patch_j = prj_patch_j.reshape(-1, (2*patch_radius+1)**2) # reshape 
            ncc_proj_j = src_patches[j].reshape(1,-1) @ prj_patch_j.T / (2*patch_radius+1)**2 # ncc 

            #print('ncc_base_j \t = ', ncc_base_j)
            #print('ncc_proj_j.max() = ', ncc_proj_j.max()) 
            
            if ncc_proj_j.shape == (0,):
                pixel = target_kpts[:,j] 
                #print(f'{pixel}: shape 0')
                if 0:
                    print('target_kpts = ', target_kpts.T[j]) 
                    print('Oj, Tj = ', Oj,Tj) 
                    npprint(f'target_patches[{Oj[1]}:{Tj[1]}, {Oj[0]}:{Tj[1]}]', target_patches[Oj[1]:Tj[1], Oj[0]:Tj[1]])
                    npprint(ncc_proj_j, 'ncc_proj')
                    #npprint(prj_patch_j, 'projectd_patch')
                    #print(target_patches.shape) 
                    #print(ncc_proj.shape)
                    #npprint(kpt_patches, 'kpt_patches')
            
            else: 
                tgt_idj = target_kpts[:,j]
                id_j = ncc_proj_j.argmax() 
                delta_j = np.array([id_j  % width_j, id_j // width_j])
                prj_idj = Oj + delta_j
                dj = np.sqrt(np.sum((tgt_idj - prj_idj)**2))
                #print(dj)

                if dj < 6:
                    pixel = tgt_idj
                
                else:
                    if ncc_base_j > 0.9:
                        pixel = tgt_idj

                    elif ncc_proj_j.max()>0.85:
                        pixel = prj_idj
                         
                    else:
                        pixel = tgt_idj 
                #print(f'{pixel}: shape non0')

            '''
            elif ncc_base_j >= ncc_proj_j.max() * 1.1:
                pixel = target_kpts[:,j] 
                print(f'{pixel}: ncc right') 
                #xj = int(target_kpts[0,j].round()); yj = int(target_kpts[1,j].round())
                #print('target is the most relieable') 

            else:
                id = ncc_proj_j.argmax() 
                dx = id  % width_j; dy = id // width_j
                xj = x0 + dx; yj = y0 + dy
                pixel = np.array([xj,yj])
                prj_j = np.array([xj,yj]) 
                print(f'{pixel}: patch')

                if 0:
                    vprint(id, 'id') 
                    npprint(target_kpts[:,j], f'target_kpts[:,{j}]')
                    print(x0,x1,y0,y1)
                    npprint(ncc_proj_j, 'ncc_proj_j')
                    npprint(target_patches[y0:y1, x0:x1], 'target_patches[y0:y1, x0:x1]')
            '''
            
            polish_list[i]['keypoints'][j] = dict({'id': j, 'pixel': pixel.T.tolist()})
        polish_kpts = getNumpyKpts(polish_list[i]) 
        # ------------------------------------------------------------------------------------------------

        if 0:
            # kpts image 
            image00 = impoints(source_image, source_kpts.T) 
            image01 = impoints(target_image, polish_kpts.T) 
            image02 = impoints(target_image, target_kpts.T, color=(0,0,255))
            image02 = impoints(image02, projected_kpts.T, color=(0,255,0))
            image0 = np.hstack((image00, image01, image02))

            # grid image 
            projected_grid = (A2 @ homogenous(source_grid))[:-1]
            err = np.sqrt(np.sum((projected_grid - target_grid)**2,axis=0))
            polish_grid = copy.deepcopy(target_grid) 
            polish_grid[:,err > thr2] = projected_grid[:,err > thr2] 

            image10 = impoints(source_image, source_grid.T) 
            image11 = impoints(target_image, polish_grid.T) 
            image12 = impoints(target_image, target_grid.T, color=(0,0,255))
            image12 = impoints(image12, projected_grid.T, color=(0,255,0)) 
            image1 = np.hstack((image10, image11, image12)) 

            # final image 
            image = np.vstack((image0, image1))
            imshow(image, f'{i}_{filename}')

    t1 = time.time(); print(f'***computation ends*** {t1-t0:.2f} sec') 
    jssave(polish_list, os.path.join(polish_dir, f'pr{patch_radius}_all{thr1}_part{thr2}_{patch_radius}.json')) 

def polishLinearly(args, patch_radius=9): 
    title(f'polish linearly: patch_radius = {patch_radius}') 

    source_list = jsread(os.path.join(args['data_dir'], 'keypoints_source.json')) 
    #target_list = jsread(os.path.join(args['ans_dir'], 'ncc_target_list', f'pr{patch_radius}.json'))
    target_list = jsread(os.path.join(args['ans_dir'], f'test_pr{patch_radius}_202112041546.json'))
    grids_list = jsread(os.path.join(args['ans_dir'], 'ncc_grids_list', f'pr{patch_radius}.json'))
    #grids_list = npload(os.path.join(args['ans_dir'], 'ncc_grids_list', f'pr{patch_radius}.npz'))

    thr1 = 9; thr2 = 6 # best 
    polish_list = copy.deepcopy(target_list) 
    polish_dir = os.path.join(args['ans_dir'], 'polish')
    if not os.path.exists(polish_dir): os.makedirs(polish_dir) 
    
    for i in range(len(source_list)):
        source = source_list[i] 
        filename = source['filename'] 

        source_kpts = getNumpyKpts(source_list[i])
        target_kpts = getNumpyKpts(target_list[i])

        source_grid = np.array(grids_list[i]['source_grid']).T
        target_grid = np.array(grids_list[i]['target_grid']).T 

        A1 = DirectLinearTransformation(source_grid, target_grid) 
        projected_grid = (A1 @ homogenous(source_grid))[:-1]
        err1 = np.sqrt(np.sum((projected_grid - target_grid)**2, axis=0)) 
        ids1 = (err1 < thr1) 

        A2 = DirectLinearTransformation(source_grid[:,ids1], target_grid[:,ids1])
        projected_kpts = (A2 @ homogenous(source_kpts))[:-1]
        err2 = np.sqrt(np.sum((projected_kpts - target_kpts)**2, axis=0)) 
        ids2 = (err2 > thr2)

        polish_kpts = copy.deepcopy(target_kpts) 
        polish_kpts[:,ids2] = projected_kpts[:,ids2]
        polish_list[i]['keypoints'] = toKptList(polish_kpts)

        if 1:
            source_image = imread(os.path.join(args['data_dir'], 'images_source', filename)) 
            target_image = imread(os.path.join(args['data_dir'], 'images_target', filename))

            # kpts image 
            image00 = impoints(source_image, source_kpts.T) 
            image01 = impoints(target_image, polish_kpts.T) 
            image02 = impoints(target_image, target_kpts.T, color=(0,0,255))
            image02 = impoints(image02, projected_kpts.T, color=(0,255,0))
            image0 = np.hstack((image00, image01, image02))

            # grid image 
            projected_grid = (A2 @ homogenous(source_grid))[:-1]
            err = np.sqrt(np.sum((projected_grid - target_grid)**2,axis=0))
            polish_grid = copy.deepcopy(target_grid) 
            polish_grid[:,err > thr2] = projected_grid[:,err > thr2] 

            image10 = impoints(source_image, source_grid.T) 
            image11 = impoints(target_image, polish_grid.T) 
            image12 = impoints(target_image, target_grid.T, color=(0,0,255))
            image12 = impoints(image12, projected_grid.T, color=(0,255,0)) 
            image1 = np.hstack((image10, image11, image12)) 

            # final image 
            image = np.vstack((image0, image1))
            imshow(image, f'{i}_filename') 

    jssave(polish_list, os.path.join(polish_dir, f'pr{patch_radius}_all{thr1}_part{thr2}.json')) 

def showNccMatching(args, patch_radius=9, flag='kpts'):
    title(f'show ncc matching: {patch_radius}')  

    source_list = jsread(os.path.join(args['data_dir'], 'keypoints_source.json')) 
    target_list = jsread(os.path.join(args['ans_dir'], 'ncc_target_list', f'pr{patch_radius}.json'))
    grids_list = jsread(os.path.join(args['ans_dir'], 'ncc_grids_list', f'pr{patch_radius}.json'))
    
    for i in range(len(source_list)):
        source = source_list[i] 
        filename = source['filename'] 

        source_image = imread(os.path.join(args['data_dir'], 'images_source', filename)) 
        target_image = imread(os.path.join(args['data_dir'], 'images_target', filename))

        if 'kpts' in flag:
            source_kpts = getNumpyKpts(source_list[i]).T
            target_kpts = getNumpyKpts(target_list[i]).T 
            kpts_image = np.hstack((impoints(source_image, source_kpts), impoints(target_image, target_kpts))) 
            imshow(kpts_image, filename) 

        if 'grid' in flag:
            t0 = time.time() 
            source_grid = grids_list[i]['source_grid'] 
            target_grid = grids_list[i]['target_grid']
            print('time: ', time.time()-t0)

            grid_image = np.hstack((impoints(source_image, source_grid), impoints(target_image, target_grid))) 
            imshow(np.vstack((kpts_image, grid_image)), filename) 

def retakegrid(args, patch_radius=9):
    title(f'retake grid: patch_radius = {patch_radius}')

    grids_list_dir = os.path.join(args['ans_dir'], 'ncc_grids_list')

    grid_n = 40
    area_threshold = 70 

    source_list = jsread(os.path.join(args['data_dir'], 'keypoints_source.json')) 
    grids_list = jsread(os.path.join(args['ans_dir'], 'ncc_grids_list', f'pr{patch_radius}.json'))
    numDataSets = len(source_list) 

    grids_list = list(dict({
        'filename':source_list[i]['filename'],
        'patch_radius':patch_radius,
        'grid_n': grid_n,
        'area_threshold': area_threshold
        }) for i in range(numDataSets))
    grids_list_np = copy.deepcopy(grids_list) 

    print('***computation starts***'); t0 = time.time(); lap = 1; num_laps = 20
    for i in range(numDataSets):
        if (i+1) % int(numDataSets/num_laps)==0:
            print(f'{lap*int(100/num_laps)} % ends: {time.time()-t0:.2f} sec'); lap +=  1
        
        target_grid = grids_list[i]['target_grid']

        if len(target_grid) < 100:  
            # inputs 
            source = source_list[i]
            filename = source['filename']; name = filename.split('.')[0]
            source_image = imread(os.path.join(args['data_dir'], 'images_source', filename))
            target_image = imread(os.path.join(args['data_dir'], 'images_target', filename))

            # perform ncc matching  
            source_patches, target_patches = getPatches(source_image, target_image, patch_radius)        
            source_grid = getSourceGrid(source_image, grid_n=40, area_threshold=20) 
            target_grid = nccMatching(source_grid, source_patches, target_patches)
        
            grids_list[i]['source_grid'] = toIntList(source_grid) # source_grid.T.tolist()
            grids_list[i]['target_grid'] = toIntList(target_grid) # target_grid.T.tolist() 
            grids_list_np[i]['source_grid'] = toIntArray(source_grid) # source_grid.T.tolist()
            grids_list_np[i]['target_grid'] = toIntArray(target_grid) # target_grid.T.tolist() 

    t1 = time.time(); print(f'***computation ends*** {t1-t0:.2f} sec') 

    jssave(grids_list, os.path.join(grids_list_dir, f'pr{patch_radius}_re.json')) 
    np.savez(os.path.join(grids_list_dir, f'pr{patch_radius}_re.npz'), grids_list_np) 



def runNccMatching(args, patch_radius=9, flag='kpts'):
    title(f'run ncc matching: patch_radius = {patch_radius}')

    grid_n = 40
    area_threshold = 70 

    target_list_dir = os.path.join(args['ans_dir'], 'ncc_target_list')
    grids_list_dir = os.path.join(args['ans_dir'], 'ncc_grids_list')
    if not os.path.exists(target_list_dir):    os.makedirs(target_list_dir) 
    if not os.path.exists(grids_list_dir):     os.makedirs(grids_list_dir)  

    source_list = jsread(os.path.join(args['data_dir'], 'keypoints_source.json')) 
    numDataSets = len(source_list) 

    target_list = copy.deepcopy(source_list) 
    grids_list = list(dict({
        'filename':source_list[i]['filename'],
        'patch_radius':patch_radius,
        'grid_n': grid_n,
        'area_threshold': area_threshold
        }) for i in range(numDataSets))
    grids_list_np = copy.deepcopy(grids_list) 

    print('***computation starts***'); t0 = time.time(); lap = 1; num_laps = 20
    for i in range(numDataSets):
        if (i+1) % int(numDataSets/num_laps)==0:
            print(f'{lap*int(100/num_laps)} % ends: {time.time()-t0:.2f} sec'); lap +=  1
        
        # inputs 
        source = source_list[i]
        filename = source['filename']; name = filename.split('.')[0]
        source_image = imread(os.path.join(args['data_dir'], 'images_source', filename))
        target_image = imread(os.path.join(args['data_dir'], 'images_target', filename))

        # perform ncc matching  
        source_patches, target_patches = getPatches(source_image, target_image, patch_radius) 

        if 'kpts' in flag:
            # kpts 
            #t_i = time.time() 
            source_kpts = getNumpyKpts(source) 
            target_kpts = nccMatching(source_kpts, source_patches, target_patches) 
            #print(f'kpts matching: {time.time() - t_i:.5f}')

            target_list[i]['keypoints'] = toKptList(target_kpts) 
        
        if 'grid' in flag: 
            # grid 
            #t_i = time.time()        
            source_grid = getSourceGrid(source_image, grid_n, area_threshold) 
            target_grid = nccMatching(source_grid, source_patches, target_patches)
            #print(f'grid matching: {time.time() - t_i:.5f}')
        
            grids_list[i]['source_grid'] = toIntList(source_grid) # source_grid.T.tolist()
            grids_list[i]['target_grid'] = toIntList(target_grid) # target_grid.T.tolist() 
            grids_list_np[i]['source_grid'] = toIntArray(source_grid) # source_grid.T.tolist()
            grids_list_np[i]['target_grid'] = toIntArray(target_grid) # target_grid.T.tolist() 

    t1 = time.time(); print(f'***computation ends*** {t1-t0:.2f} sec') 

    if 'kpts' in flag: jssave(target_list, os.path.join(target_list_dir, f'pr{patch_radius}_pad.json'))
    if 'grid' in flag: jssave(grids_list, os.path.join(grids_list_dir, f'pr{patch_radius}_pad.json')) 
    if 'grid' in flag: np.savez(os.path.join(grids_list_dir, f'pr{patch_radius}_pad.npz'), grids_list_np) 

def toIntArray(np_coords):
    return np_coords.round().astype(np.uint8).T 

def toIntList(np_coords):
    return np_coords.round().astype(np.uint8).T.tolist() 

def toKptList(np_kpts): 
    return list(dict({'id': k, 'pixel': v}) for k, v in enumerate(np_kpts.T.tolist()))

def getNumpyKpts(source):
    return np.array(list(v['pixel'] for v in source['keypoints'])).T

def patchids(id,r=1,width=256):
    x = id  % width; y = id // width
    x0 = max(0, x - r); x1 = min(x + r + 1, width)
    y0 = max(0, y - r); y1 = min(y + r + 1, width)
    xx, yy = np.meshgrid(np.arange(x0,x1), np.arange(y0,y1)) 
    patch_ids = yy * width + xx 
    return patch_ids.flatten().astype(np.uint64) 

def getSourceGrid(source_gray, grid_n=40, area_threshold=70): 
    grid = createGrid(mesh=[grid_n,grid_n]) 
    area = (source_gray > area_threshold)
    source_grid = grid[:,area[grid[1],grid[0]]]
    return source_grid 

def createGrid(mesh=[20,20],shape=[256,256]):
    h, w = shape

    d0 = h - h//mesh[0]
    d1 = w - w//mesh[1] 

    #print('d0 = ', d0, 'd1 = ', d1, 'h', 'w') 
    y = np.linspace(d0,h-d0,mesh[0])
    x = np.linspace(d1,w-d1,mesh[1]) 

    xx, yy = np.meshgrid(x, y)
    grid = np.vstack((xx.flatten(), yy.flatten()))

    return grid.astype(np.uint8)

def nccMatching(source_points, source_patches, target_patches, width=256):
    patch_radius = int(0.5 * (np.sqrt(source_patches.shape[-1]) - 1))
    source_ids = source_points[1] * width + source_points[0]

    source_patches = source_patches.reshape(-1,(2*patch_radius+1)**2)[source_ids] 
    target_patches = target_patches.reshape(-1,(2*patch_radius+1)**2) 
    ncc = source_patches @ target_patches.T / source_patches.shape[1]

    target_ids = ncc.argmax(axis=1) 
    target_points = np.vstack((target_ids%width, target_ids//width)) 

    return target_points 

def getPatches(gray_source, gray_target, patch_radius, norm=True, path=True):
    height, width = gray_source.shape 
    
    patches_source = np.zeros((height, width, (2*patch_radius+1)**2)) # dtype??
    patches_target = np.zeros((height, width, (2*patch_radius+1)**2)) 
    i = 0
    for y in range(-patch_radius, patch_radius+1):
        for x in range(-patch_radius, patch_radius+1):
            # Shift the image to get different elements of the patch
            patches_source[:,:,i] = np.roll(np.roll(gray_source, -y, axis=0), -x, axis=1) 
            patches_target[:,:,i] = np.roll(np.roll(gray_target, -y, axis=0), -x, axis=1)
            i+=1
    
    if norm:
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

def homogenoust(x):
    numPoints = x.shape[1] 
    ones = torch.ones(numPoints).view(1,-1)
    return torch.cat([x, ones], axis=0)

def mse(p,y):
    return torch.sqrt(torch.sum((p - y)**2, axis=0)).mean()

def GradientMethod(points_source, points_target, A=None):
     
    x = torch.tensor(points_source).type(torch.float32)
    y = torch.tensor(points_target).type(torch.float32) 

    if A.all()==None:
        A = torch.zeros(2,3,requies_grad=True).type(torch.float32) 
    else:
        A = torch.tensor(A[:2], requires_grad=True, dtype=torch.float32)

    lr = 1e-5 
    losses = [] 

    for i in range(3000):
        p = torch.matmul(A, homogenoust(x)) 
        loss = mse(p,y) 
        loss.backward() 

        with torch.no_grad():
            A -= A.grad * lr 
            A.grad.zero_() 
            
        losses.append(loss) 
    
    Anp = np.vstack((A.detach().numpy(), np.array([0,0,1]))) 

    return Anp


def DirectLinearTransformation(points_source, points_target): # y = Ax 
    # ----------------------------------------------------------------
    # Input:
    #   - points_source <(2,numPoints),np.array>: 2D coords of original points 
    #   - points_target <(2,numPoints),np.array>: 2D coords of projected points  
    # Output:
    #   - A <(3,3),np.array>: a 3x3 transformation matrix
    # ----------------------------------------------------------------

    numPoints = points_source.shape[1] 

    P = np.zeros((2*numPoints,6)) 
    P[0::2,:3] = homogenous(points_source).T 
    P[1::2,3:] = homogenous(points_source).T 
    Q = points_target.T.flatten() 
    
    a = np.linalg.pinv(P.T @ P) @ P.T @ Q 
    A = np.vstack((a.reshape(2,3), np.array([0,0,1]))) 

    return A 

def quadratic(P):
    # ----------------------------------------------------------------
    # Input:
    #   - P <n_dim, numPoints>: n-D coords of points 
    # Output:
    #   - Ph <n_dim+1, numPoints: quadratic components 
    # ----------------------------------------------------------------
    numPoints = P.shape[1] 
    out = np.zeros((3,numPoints))

    out[0] = P[0]**2
    out[1] = P[1]**2
    out[2] = P[0]*P[1] 
    
    return out 
    
def homogenous(P):
    # ----------------------------------------------------------------
    # Input:
    #   - P <n_dim, numPoints>: n-D coords of points 
    # Output:
    #   - Ph <n_dim+1, numPoints: homogenous coords of input points 
    # ----------------------------------------------------------------
    ones = np.ones(P.shape[1]).reshape(1,-1)
    return np.vstack((P,ones))

def imresults(source_path, target_path, data_dir):
    source_list = jsread(source_path) # sort 
    target_list = jsread(target_path) # sort 

    numDataSets = len(source_list)
    imgsize = source_list[0]['image_size'] 

    for i in range(numDataSets):
        filename = source_list[i]['filename']
        source_file = os.path.join(data_dir, 'images_source', filename)
        target_file = os.path.join(data_dir, 'images_target', filename)

        source_image = imread(source_file, imgsize)
        target_image = imread(target_file, imgsize)

        source_kpts = getNumpyKpts(source_list[i]) 
        target_kpts = getNumpyKpts(target_list[i])
        source_image = impoints(source_image, source_kpts)
        target_image = impoints(target_image, target_kpts) 

        image = np.hstack((source_image, target_image))
        imshow(image, filename) 

def impoints(img, pts, r=2, color=(0,0,255), thickness=-1):
    img = imrgb(img) 
    for pt in pts:
        img = cv2.circle(img, (round(pt[0]), round(pt[1])), r, color, thickness)
    return img 

def imrgb(img):
    if len(img.shape) == 2:
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255 # ok ?? 
        img = img.astype(np.uint8) 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 
    return img 

def imshow(img, name="img", save=False):
    if save: cv2.imwrite(f'../../results/{name}.png', img) 

    cv2.imshow(name, img)
    while True: 
        if cv2.waitKey(1)==ord('q'): break
    cv2.destroyAllWindows()

def imread(filepath, image_size=[256,256]):
    f = open(filepath, 'rb')
    arr = np.fromfile(f, dtype=np.float64, count=image_size[0]*image_size[1])
    image = arr.reshape((image_size[0], image_size[0]))
    return image 

def npload(filepath):
    return np.load(filepath, allow_pickle=True)['arr_0']

def jssave(pydata, filepath):
    json.dump(pydata, open(filepath, 'w'))

def jsread(filepath):
    return json.load(open(filepath, 'r'))  

def tprint(tensor, name='tensor'):
    print(f'\n {name}{tensor.shape, tensor.dtype} = \n', tensor) 

def npprint(array, name='array'):
    print(f'\n {name}{array.shape, array.dtype} = \n', array)  

def vprint(v, name='v'):
    print(f'\n {name}{type(v)} = ')
    pprint(v)  

def title(name, length=100):
    print('-'*length) 
    print('\t'+name) 
    print('-'*length) 

def show_loaded_data(args):
    #-----------------------------------------------------------
    # show loaded data 
    #-----------------------------------------------------------
    title('show loaded data')

    n = 2
    source_list = jsread(os.path.join(args['data_dir'],'keypoints_source.json')) 
    numDataSets = len(source_list)
    print() 
    print('numDataSets = ', numDataSets)
    print()
    print(f'the first {n} datasets')
    vprint(source_list[:n], f'source_data[:{n}]') 

    for i in range(numDataSets): 
        # ---------------------------------------------------------------
        #   show selected sample data
        # ---------------------------------------------------------------

        # select sample 
        sample = source_list[i]  
        vprint(sample, 'sample')

        # extract info of the sample 
        filename = sample['filename'] 
        source_image_path = os.path.join(args['data_dir'], 'images_source', filename)
        target_image_path = os.path.join(args['data_dir'], 'images_target', filename)
        image_size = sample['image_size'] 

        # image 
        source_image = imread(source_image_path, image_size)
        target_image = imread(target_image_path, image_size) 

        # key points 
        kpts_list = sample['keypoints']
        kpts_xy = np.array(list(v['pixel'] for v in kpts_list)).T 

        # show images with keypoints on the souce image 
        source_image = impoints(source_image, kpts_xy.T) 
        #target_image = impoints(target_image, [[0,0]])
        target_image = imrgb(target_image)
        img = np.hstack((source_image, target_image))  
        imshow(img, f'{i}_{filename}') 


        print('source_image_path = ', source_image_path)
        print('target_image_path = ', target_image_path)  
        vprint(image_size, 'image_size')

        vprint(kpts_list, 'kpts_list') 
        npprint(kpts_xy, 'kpts_xy') 


def showSIFT(img1, img2, kpts1):
    tform = transform.AffineTransform(scale=(1.3, 1.1), rotation=0.5,
                                  translation=(0, -200))
    img3 = transform.warp(img1, tform)

    descriptor_extractor = SIFT()

    descriptor_extractor.detect_and_extract(img1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img3)
    keypoints3 = descriptor_extractor.keypoints
    descriptors3 = descriptor_extractor.descriptors

    matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=0.6,
                                cross_check=True)
    matches13 = match_descriptors(descriptors1, descriptors3, max_ratio=0.6,
                                cross_check=True)
    
    npprint(keypoints1, 'keypoints1')
    npprint(keypoints2, 'keypoints2') 
    npprint(matches12, 'matches12')

    d = scipy.spatial.distance.cdist(kpts1, keypoints1, metric='euclidean')
    ids = d.argmin(axis=1) 
    npprint(kpts1, 'kpts1') 
    npprint(keypoints1, 'keypoints1') 
    npprint(keypoints1[ids], 'keypoints1[ids')
    npprint(descriptors1, 'descriptors1') 
    npprint(descriptors1[ids], 'descriptors1[ids]')

    matches_kpts = match_descriptors(descriptors1[ids], descriptors2, max_ratio=1, cross_check=True)
    npprint(matches_kpts, 'matches_kpts') 

    image0 = impoints(img1, kpts1)
    image0 = impoints(image0, keypoints1[ids], color=(0,255,0))
    image1 = impoints(img2, keypoints2[matches_kpts.T[1]])
    imshow(np.hstack((image0, image1)))

    if 0:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))

        plt.gray()

        plot_matches(ax[0, 0], img1, img2, keypoints1, keypoints2, matches12)
        ax[0, 0].axis('off')
        ax[0, 0].set_title("Original Image vs. Flipped Image\n"
                        "(all keypoints and matches)")

        plot_matches(ax[1, 0], img1, img3, keypoints1, keypoints3, matches13)
        ax[1, 0].axis('off')
        ax[1, 0].set_title("Original Image vs. Transformed Image\n"
                        "(all keypoints and matches)")

        plot_matches(ax[0, 1], img1, img2, keypoints1, keypoints2, matches12[::15],
                    only_matches=True)
        ax[0, 1].axis('off')
        ax[0, 1].set_title("Original Image vs. Flipped Image\n"
                        "(subset of matches for visibility)")

        plot_matches(ax[1, 1], img1, img3, keypoints1, keypoints3, matches13[::15],
                    only_matches=True)
        ax[1, 1].axis('off')
        ax[1, 1].set_title("Original Image vs. Transformed Image\n"
                        "(subset of matches for visibility)")

        plt.tight_layout()
        plt.show()

if __name__=='__main__':
    main() 
