import itk # https://github.com/InsightSoftwareConsortium/ITKElastix

import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import os 
import json 
import copy
from time import time 
from glob import glob 
#from datetime import datetime 

import algQ2 as alg 

DIR_INPUT = '../../input/Q2' 
DIR_OUTPUT = '../../output/Q2'
DIR_EXTRA = os.path.join(DIR_OUTPUT, 'extra')
DIR_ANSWER = os.path.join(DIR_OUTPUT, 'answer') 

for dir_path in [DIR_INPUT, DIR_OUTPUT, DIR_EXTRA, DIR_ANSWER]:
    if not os.path.exists(dir_path): os.makedirs(dir_path) 

output_dir = './output_dir'
if not os.path.exists(output_dir): os.makedirs(output_dir) 
point_transformed_dir = os.path.join(output_dir, 'point_transformed')
if not os.path.exists(point_transformed_dir): os.makedirs(point_transformed_dir) 

# ===========================================================
#   inputs 
# ===========================================================
alg.title('basic info'); debug0 = 1

source_path_list = sorted(glob(os.path.join(DIR_INPUT, 'images_source', '*.raw')))
target_path_list = sorted(glob(os.path.join(DIR_INPUT, 'images_target', '*.raw')))
source_data_list = alg.jsread(os.path.join(DIR_INPUT, 'keypoints_source.json'))  
source_data_list = sorted(source_data_list, key=lambda x: x['filename'])
target_data_list = copy.deepcopy(source_data_list) 
numDataSets = len(source_data_list) 
target_data_path = os.path.join(DIR_ANSWER, 'submission_itk.json')

if debug0:
    alg.pyprint(source_path_list, f'source_path_list') 
    alg.pyprint(target_path_list, f'target_path_list')  
    
print('numDataSets = ', numDataSets) 
print('the result will be saved in: ', target_data_path) 

# ===========================================================
#   perform comuputation  
# ===========================================================
alg.title('perform computation'); debug1 = 1

manager = alg.Manager(time(), size=numDataSets, num_laps=100)
for i in range(numDataSets):
    manager.step(i, time()) 

    # -----------------------------------------------------------
    #   input 
    # -----------------------------------------------------------
    sample = source_data_list[i] 
    filename = sample['filename']
    keypoints = sample['keypoints'] 
    source_kpts = alg.kptList2Array(keypoints) 
    source_path = source_path_list[i] 
    target_path = target_path_list[i] 
    source_image = alg.imread(source_path)
    target_image = alg.imread(target_path) 
    source_kpts = alg.kptList2Array(keypoints) 
    numKpts = source_kpts.shape[1] 

    assert filename in source_path and filename in target_path, \
        f'filename={filename} not in source_path ({source_path}) or target_path ({target_path})' 
    assert source_image.shape==target_image.shape, \
        f'source and target images have different shapes: \
        source: {source_image.shape} and target{target_image.shape}'
    height, width = source_image.shape

    if debug1:
        alg.subtitle(f'{i}_{filename}') 
        alg.subsubtitle('input') 
        print('source_path = ', source_path) 
        print('target_path = ', target_path) 
        print('source_image.shape == target_image.shape is True')
        print('image shape (height, width) = ', height, width) 
        alg.pyprint(keypoints, 'keypoints') 
        alg.npprint(source_kpts, 'source_kpts') 
        alg.imshow(alg.imrgb(np.hstack((source_image, target_image))), filename, save=True) 

    # -----------------------------------------------------------
    #   preprocess 
    # -----------------------------------------------------------
    fixed_image = source_image.astype(np.float32) 
    moving_image = target_image.astype(np.float32) 

    # -----------------------------------------------------------
    #   optimization with B-Spline model
    # -----------------------------------------------------------
    # Forward Parameter Map
    parameter_object = itk.ParameterObject.New()
    parameter_map_rigid = parameter_object.GetDefaultParameterMap('rigid')
    parameter_object.AddParameterMap(parameter_map_rigid)
    parameter_map_affine = parameter_object.GetDefaultParameterMap('affine')
    parameter_object.AddParameterMap(parameter_map_affine)
    parameter_map_bspline = parameter_object.GetDefaultParameterMap('bspline')
    parameter_object.AddParameterMap(parameter_map_bspline)

    # Registration
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image,
        parameter_object=parameter_object,
        output_directory=output_dir
    )

    # Create points
    #fixed_points = np.random.random([10,2])*200 + 30
    fixed_points = copy.deepcopy(source_kpts.T) 
    np.savetxt("fixed_points.txt", fixed_points, fmt = "%.5f")

    # Modify the file
    with open("fixed_points.txt", 'r') as f:
        l = f.readlines()

    l.insert(0, 'point\n')
    l.insert(1, f'{numKpts}\n')

    with open("fixed_points.txt", 'w') as f:
        f.writelines(l)

    result_image = itk.transformix_filter(
        moving_image, result_transform_parameters,
        fixed_point_set_file_name='fixed_points.txt',
        output_directory=point_transformed_dir)

    result_points = np.loadtxt(
        os.path.join(point_transformed_dir,'outputpoints.txt'), 
        dtype='str')[:,27:29].astype('float64')
    
    # -----------------------------------------------------------
    #   write the projected keypoints into list 
    # -----------------------------------------------------------
    target_data_list[i]['keypoints'] = alg.kptArray2List(result_points.T) 

    if debug1: 
        print('result_transform_parameters = \n', result_transform_parameters)
        alg.pyprint(fixed_points, 'source_kpts') 
        alg.pyprint(result_points, 'target_kpts') 
        image0 = alg.impoints(source_image, fixed_points) 
        image1 = alg.impoints(target_image, result_points, color=(0,255,0)) 
        alg.imshow(np.hstack((image0, image1)),f'Q2_{filename}_itk', save=True) 

manager.end(time()) 

# -----------------------------------------------------------
#   save the projected kpts  
# -----------------------------------------------------------
alg.jssave(target_data_list, target_data_path)     
