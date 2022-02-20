import cv2 
import json
import numpy as np 
import matplotlib.pyplot as plt 

import os 
import copy 
import time 
from pprint import pprint 
from datetime import datetime

DIR_INPUT = '../../input/Q1' 
DIR_OUTPUT = '../../output/Q1' 
for dir_path in [DIR_INPUT, DIR_OUTPUT]:
    if not os.path.exists(dir_path): os.makedirs(dir_path)  

def main():    
    if 1:
        #-----------------------------------------------------------
        # solve all 
        #-----------------------------------------------------------
        title('solve all')

        # input 
        filename = os.path.join(DIR_INPUT, 'points.json')
        points = jsread(filename) 

        # solve 
        ans = copy.deepcopy(points) 
        ans['Q1_1'] = solve_Q1_1(points['Q1_1']) 
        ans['Q1_2'] = solve_Q1_2(points['Q1_2']) 
        ans['Q1_3'] = solve_Q1_3(points['Q1_3']) 

        # save 
        now = datetime.now().strftime("%Y%m%d%H%M") 
        ansname = os.path.join(DIR_OUTPUT, f'ans_{now}.json') 
        jssave(ans, ansname)
    
    if 0:
        #-----------------------------------------------------------
        # 与えられた.jsonファイルの読み込み
        #-----------------------------------------------------------
        title('test: load a json file') 

        filename = os.path.join(DIR_INPUT, 'points.json') 
        points = jsread(filename) 
        vprint(filename, 'filename') 
        vprint(points, 'points') 
    
    if 0:
        #-----------------------------------------------------------
        # 任意のdict型データを生成し、.jsonとして保存し中身を確認
        #-----------------------------------------------------------
        title('test: save json')

        filename = os.path.join(DIR_OUTPUT, 'sample_data.json') 
        data_tobe_saved = {'a': [1,2,3], 'b':[4,5,6], 'c':[7,8,9]} 
        jssave(data_tobe_saved, filename) 
        data_read = jsread(filename) 

        vprint(filename, 'filename') 
        vprint(data_tobe_saved, 'data_tobe_saves') 
        vprint(data_read, 'data read') 
    
    if 1:
        #-----------------------------------------------------------
        # points.json中のQ1_1　に該当する座標について
        # 原点(0,0)を中心として、反時計回りに30°回転させて得られる座標を提出
        #-----------------------------------------------------------
        title('test: solve Q1_1') 

        filename = os.path.join(DIR_INPUT, 'points.json')
        points = jsread(filename) 
        points_Q1_1 = solve_Q1_1(points['Q1_1']) 
        imcompare2Dpoints(points['Q1_1'], points_Q1_1, 'source', 'target', title='Q1_1') 
    
    if 1:
        #-----------------------------------------------------------
        # points.json中のQ1_2　に該当する座標について
        # 点(128,128)からの距離を1.28倍に拡大させた後の座標を提出
        #-----------------------------------------------------------
        title('test: solve Q1_2') 

        filename = os.path.join(DIR_INPUT, 'points.json') 
        points = jsread(filename) 
        points_Q1_2 = solve_Q1_2(points['Q1_2']) 
        imcompare2Dpoints(points['Q1_2'], points_Q1_2, 'source', 'target',  title='Q1_2') 
    
    if 1:
        #-----------------------------------------------------------
        # points.json中のQ1_3　に該当する座標について
        # 所定の変換手順を行なった後の座標を提出
        #-----------------------------------------------------------
        title('test: solve Q1_3') 

        filename = os.path.join(DIR_INPUT, 'points.json') 
        points = jsread(filename) 
        points_Q1_3 = solve_Q1_3(points['Q1_3']) 
        imcompare2Dpoints(points['Q1_3'], points_Q1_3, 'source', 'target',  title='Q1_3') 

def solve_Q1_1(points):
    #-----------------------------------------------------------------
    # Q1_1: 原点(0,0)を中心として、反時計回りに30°回転させて得られる座標を得る
    #
    # Input:
    #   - points(numPoints, 2)<list>: before pi/6 rotation around (0,0)
    # Output: 
    #   - points(numPoints, 2)<list>: after pi/6 rotation around (0,0) 
    #-----------------------------------------------------------------

    points = homogenous(np.array(points).T)

    points = rot2D(30) @ points 

    return points[:-1].T.tolist() 

def solve_Q1_2(points):
    #-----------------------------------------------------------------
    # Q1_2: 点(128,128)からの距離を1.28倍に拡大させた後の座標を得る
    #
    # Input:
    #   - points(numPoints, 2)<list>: before x1.28 from (128, 128)
    # Output: 
    #   - points(numPoints, 2)<list>: after x1.28 from (128, 128) 
    #-----------------------------------------------------------------

    points = homogenous(np.array(points).T) 

    points = translation2D(128,128) @ mag2D(1.28) @ translation2D(-128,-128) @ points 

    return points[:-1].T.tolist() 

def solve_Q1_3(points):
    #-----------------------------------------------------------------
    # Q1_3: 以下の操作を上から順に施したものを提出
    #   - 点(128,128)を中心として反時計回りに12度回転
    #   - 点(128,128)からの距離を0.8倍に縮小
    #   - x方向に12, y方向に8移動
    #
    # Input:
    #   - points(numPoints, 2)<list>: before the operations 
    # Output: 
    #   - points(numPoints, 2)<list>: after the operations 
    #-----------------------------------------------------------------

    points = homogenous(np.array(points).T)

    points = translation2D(128,128) @ rot2D(12) @ translation2D(-128,-128) @ points 

    points = translation2D(128,128) @ mag2D(0.8) @ translation2D(-128,-128) @ points 

    points = translation2D(12,8) @ points 

    return points[:-1].T.tolist() 

def translation2D(x=-128,y=-128):
    #-----------------------------------------------------------------
    # Input:
    #   - trans(2)<tuple>: 2D translation (x, y) 
    # Output:
    #   - matrix(3, 3)<np.ndarray>: 2D translation matrix 
    #-----------------------------------------------------------------
    
    T = np.array([ 
        [1, 0, x],
        [0, 1, y],
        [0, 0,    1]
    ])

    return T 

def mag2D(factor=1.28):
    #-----------------------------------------------------------------
    # Input:
    #   - facotr(1)<float>: magnification factor 
    # Output:
    #   - matrix(3, 3)<np.ndarray>: 2D magnification matrix 
    #-----------------------------------------------------------------
    
    M = np.array([ 
        [factor, 0, 0],
        [0, factor, 0],
        [0,     0,  1]])
    
    return M
    

def rot2D(theta=30):
    #-----------------------------------------------------------------
    # Input:
    #   - theta<float>: rotation angle [degree] (+ for counterclockwise)
    # Output:
    #   - - matrix(3, 3)<np.ndarray>: 2D rotation matrix 
    #-----------------------------------------------------------------
    
    theta = deg2rad(theta) 
    c = np.cos(theta) 
    s = np.sin(theta) 

    R = np.array([ 
        [c, -s,  0],
        [s,  c,  0],
        [0,  0,  1]
    ])

    return R  

def homogenous(points):
    #-----------------------------------------------------------------
    # Input:
    #   - points(2, numPoints)<np.ndarray>:  2D coords 
    # Output: 
    #   - points(2+1, numPoints)<np.ndarray>: homogenous coods 
    #-----------------------------------------------------------------
    ones = np.ones(points.shape[1], dtype=points.dtype) 
    return np.vstack( (points, ones ) )

def deg2rad(theta):
    return theta * np.pi/180 

def imcompare2Dpoints(points0, points1, name0='data0', name1='data1', title='Q1'):
    #-----------------------------------------------------------------
    # Input:
    #   - points(numPoints, 2)<list>: 2D coords 
    #   - points(numPoints, 2)<list>: 2D coords 
    #-----------------------------------------------------------------
    points0 = np.array(points0).T 
    points1 = np.array(points1).T 

    xmax = max(np.max(np.abs(points0[0])), np.max(np.abs(points1[0])))*1.1
    ymax = max(np.max(np.abs(points0[1])), np.max(np.abs(points1[1])))*1.1

    plt.scatter(points0[0], points0[1], label=name0) 
    plt.scatter(points1[0], points1[1], label=name1) 

    #plt.xlim(-xmax, xmax); plt.xlabel('X')
    #plt.ylim(-ymax, ymax); plt.ylabel('Y') 

    plt.xlim(-30, 250); plt.xlabel('X')
    plt.ylim(-30, 250); plt.ylabel('Y') 
    plt.grid() 
    plt.legend() 
    plt.title(title) 
    plt.show() 
    path = os.path.join(DIR_OUTPUT, f'{title}_fixed_title.png') 
    plt.savefig(path)

def nameof(var):
    for k,v in globals().items():
        if id(v) == id(var): name=k
    return name

def jssave(data, filename):
    json.dump(data, open(filename, 'w'))
    print(f'\n data saved in {filename}')

def jsread(filename):
    return json.load(open(filename))

def npprint(array, name='array'):
    print(f'\n {name}{array.shape, array.dtype} = \n', array)  

def vprint(v, name='v'):
    print(f'\n {name}{type(v)} = ')
    pprint(v)  

def title(name, length=100):
    print('-'*length) 
    print(f'\t {name}') 
    print('-'*length) 

if __name__=="__main__":
    main()