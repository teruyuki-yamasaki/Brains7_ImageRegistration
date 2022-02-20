import os 
import copy
import numpy as np 
from time import time 
from glob import glob 

import algQ3 as alg 

DIR_INPUT = '../../input/Q3' 
DIR_OUTPUT = '../../output/Q3'
DIR_EXTRA = os.path.join(DIR_OUTPUT, 'extra')
DIR_ANSWER = os.path.join(DIR_OUTPUT, 'answer') 

for dir_path in [DIR_INPUT, DIR_OUTPUT, DIR_EXTRA, DIR_ANSWER]:
    if not os.path.exists(dir_path): os.makedirs(dir_path)  

def main():
    # ===========================================================
    #   inputs 
    # ===========================================================
    alg.title('basic info'); debug0 = 0

    source_path_list = sorted(glob(os.path.join(DIR_INPUT, 'volumes_source', '*.npz')))
    target_path_list = sorted(glob(os.path.join(DIR_INPUT, 'volumes_target', '*.npz')))
    source_data_list = alg.jsread(os.path.join(DIR_INPUT, 'keypoints_source.json'))  
    source_data_list = sorted(source_data_list, key=lambda x: x['IXI_ID'])
    target_data_list = copy.deepcopy(source_data_list) 
    num_data_sets = len(source_data_list) 

    perms = ['012','021','102','120','201','210'] # 102 inv gives the answer
    target_data_lists = list(copy.deepcopy(source_data_list) for i in range(len(perms)))
    target_hinv_lists = list(copy.deepcopy(source_data_list) for i in range(len(perms)))
    print('numDataSets = ', num_data_sets) 

    if debug0:
        alg.pyprint(source_path_list, f'source_path_list') 
        alg.pyprint(target_path_list, f'target_path_list')  
        alg.pyprint(num_data_sets, 'num_data_sets') 
    
    # ===========================================================
    #   perform comuputation  
    # ===========================================================
    alg.title('perform computation'); debug1 = [0,0,0,1] #[0,0,0,0]#

    manager = alg.Manager(time(), size=num_data_sets, num_laps=20)
    for i in range(num_data_sets):
        manager.step(i,time())
        
        # -----------------------------------------------------------
        #   input 
        # -----------------------------------------------------------
        sample = source_data_list[i] 
        Id = sample['IXI_ID'] 
        filename = f'{Id}*.npz'
        keypoints = sample['keypoints'] 
        source_kpts = alg.kptList2Array(keypoints) 
        source_path = source_path_list[i] 
        target_path = target_path_list[i] 
        source_volume = alg.vmread(source_path) 
        target_volume = alg.vmread(target_path) 
        assert Id in source_path and Id in target_path, \
            f'source_path ({source_path}) or target_path ({target_path}) does not contain Id={Id}' 
        assert source_volume.shape==target_volume.shape, \
            f'source and target volumes have different shapes: \
            source: {source_volume.shape} and target: {target_volume.shape}'
        depth, height, width = source_volume.shape

        # calculate the gradient of each layer 
        source_grad_xy = alg.vmgrad(source_volume)
        target_grad_xy = alg.vmgrad(target_volume) 
        source_volume_grad = alg.vmnorm(np.sqrt(source_grad_xy[0]**2 + source_grad_xy[1]**2))
        target_volume_grad = alg.vmnorm(np.sqrt(target_grad_xy[0]**2 + target_grad_xy[1]**2))

        if debug1[0]:
            alg.subtitle(f'{i}_{Id}') 
            alg.subsubtitle('input')
            print('source_path = ', source_path) 
            print('target_path = ', target_path) 
            print('source_volume.shape == target_volume.shape is True')
            print('volume shape (depth, height, width) = ', depth, height, width) 
            alg.pyprint(keypoints, 'keypoints') 
            alg.npprint(source_kpts, 'source_kpts') 
        
        if debug1[1]:
            alg.subsubtitle('volume normalization')

            alg.vmview(volumes=[source_volume, target_volume])
            z = 40
            source_image = alg.imnorm(source_volume[z]) 
            target_image = alg.imnorm(target_volume[z]) 
            alg.imshow(alg.imrgb(np.hstack((source_image, target_image))),name=f'{Id}_z{z}',save=True)

            source_Ix = alg.imnorm(source_grad_xy[0,z])
            source_Iy = alg.imnorm(source_grad_xy[1,z]) 
            target_Ix = alg.imnorm(target_grad_xy[0,z])
            target_Iy = alg.imnorm(target_grad_xy[1,z]) 
            source_IxIy = np.vstack((source_Ix, source_Iy)) 
            target_IxIy = np.vstack((target_Ix, target_Iy))
            source_grad = alg.imnorm(source_volume_grad[z]) 
            target_grad = alg.imnorm(target_volume_grad[z]) 
            source_grads = np.vstack((source_IxIy, source_grad)) 
            target_grads = np.vstack((target_IxIy, target_grad)) 
            source_images = np.vstack((source_image, source_grads))
            target_images = np.vstack((target_image, target_grads)) 
            alg.imshow(alg.imrgb(np.hstack((source_grad, target_grad))),name=f'{Id}_z{z}_grad',save=True)
            alg.imshow(alg.imrgb(np.hstack((source_IxIy, target_IxIy))),name=f'{Id}_z{z}_IxIy',save=True)
            alg.imshow(alg.imrgb(np.hstack((source_grads, target_grads))),name=f'{Id}_z{z}_grads',save=True)
            alg.imshow(alg.imrgb(np.hstack((source_images, target_images))),name=f'{Id}_z{z}_images',save=True)
            source_images_h = np.hstack((source_image, source_Ix, source_Iy, source_grad)) 
            target_images_h = np.hstack((target_image, target_Ix, target_Iy, target_grad))
            alg.imshow(alg.imrgb(np.vstack((source_images_h, target_images_h))),name=f'{Id}_z{z}_images_h',save=True)

            ims = []
            for f in (1,3,10):
                source_grad_f = alg.vmrescale(alg.vmdown(source_volume_grad,f=f),f=(f,f,f))
                target_grad_f = alg.vmrescale(alg.vmdown(target_volume_grad,f=f),f=(f,f,f))

                source_grad_fz = alg.imrgb(source_grad_f[z])[:256,:256] #alg.imrgb(source_grad_f[int(z/f)])
                target_grad_fz = alg.imrgb(target_grad_f[z])[:256,:256]
                im = np.vstack((source_grad_fz, target_grad_fz))

                alg.imshow(im, name=f'{im.shape}') 
                ims.append(im)  
            

                #alg.npprint(source_grad_f, 'source_grad_f')
                #alg.imshow(np.hstack((alg.imrgb(source_grad_f[int(z/f)]),alg.imrgb(target_grad_f[int(z/f)]))),name=f'grad{f}',save=True) 
            
            im = np.concatenate(ims,axis=1) 
            alg.imshow(im, name=f'{Id}_z{z}_downsampled', save=True)
 
            # normalize volumes 
            source_volume_nm = alg.vmnorm(source_volume) 
            target_volume_nm = alg.vmnorm(target_volume, minshift=True) 

            alg.vmhist(alg.vmread(source_path), alg.vmread(target_path), name='without_normalization')
            alg.vmhist(source_volume_nm, target_volume_nm, name='with_normalization')

            alg.vmshow(alg.vmcat(alg.vmread(source_path), alg.vmread(target_path)), name=f'{i}_{Id}') 
            alg.vmshow(alg.vmcat(source_volume_nm, target_volume_nm), name=f'{i}_{Id}_nm')
        
        if debug1[2]:
            alg.subsubtitle('image gradient')

            alg.vmhist(
                np.sqrt(source_grad_xy[0]**2 + source_grad_xy[1]**2), 
                np.sqrt(target_grad_xy[0]**2 + target_grad_xy[1]**2), 
                name='imgrad_volumes_without_normalization')

            alg.vmhist(
                source_volume_grad, 
                target_volume_grad, 
                name='imgrad_volumes_with_normalization')
            
            Ix = np.concatenate([source_grad_xy[0],source_grad_xy[1]],axis=1)
            Iy = np.concatenate([target_grad_xy[0],target_grad_xy[1]],axis=1)
            IxIy = np.concatenate([Ix,Iy],axis=2) 
            Ig = np.concatenate([source_volume_grad, target_volume_grad],axis=2)
            alg.vmshow(np.concatenate([alg.vmnorm(IxIy),alg.vmnorm(Ig)],axis=1), name=f'{i}_{Id}_imgrad') 
            
        # -------------------------------------------------------------------------
        #   optimization with Affine Transformation model and gradient voluems 
        # -------------------------------------------------------------------------
        T = (2,1,0); invT = (2,1,0); factors=(10,3)
        
        # optimization 
        x0 = alg.Affine2coeff3d(np.eye(4)) # begin with the identity matrix 
        x_temp = copy.deepcopy(x0) 
        for factor in factors:
            x_temp = alg.optAffineModel(
                x_temp,
                alg.AffineModel3d, 
                source_volume_grad.transpose(*T), #!!! transposed: zyx -> xyz 
                target_volume_grad.transpose(*T), #!!! transposed: zyx -> xyz 
                f=factor
            )
        M_temp = alg.coeff2Affine3d(x_temp)
        
        if debug1[3]:
            alg.subsubtitle('optimized')

            # normalize volumes for visualization 
            source_volume = alg.vmnorm(source_volume) 
            target_volume = alg.vmnorm(target_volume, minshift=True) 
            print('volumes are normalized for visualization') 

            alg.npprint(M_temp, 'M_opt') 
            warped_volume = alg.AffineModel3d(x_temp, source_volume.transpose(*T)).transpose(*invT) #!!! zyx -> xyz -> zyx
            warped_volume = alg.vmnorm(warped_volume) 
            target_kpts = (M_temp @ alg.homog(source_kpts))[:-1] 
            kpt_image = getKptCrossSections(source_kpts, target_kpts, 
                M_temp, source_volume, warped_volume, target_volume)

            alg.imshow(kpt_image, f'{Id}_sample_kpts', save=True)
            alg.vmview(volumes=[source_volume, warped_volume, target_volume], M=M_temp)
            alg.vmshow(alg.vmcat(source_volume, warped_volume, target_volume), name=f'{i}_{Id}_result')

        # -------------------------------------------------------------------------
        #   Coordinate Adjustment 
        # -------------------------------------------------------------------------
        '''
        Note that the estimated Affine Transform matrix 'M_temp' is calculated
        in the coordinate system of volume data, which is set arbitrarily. 
        In order to apply this transform to keypoint estimation, we need to adjust 
        the transform to the keypoint coordinate system.
        '''
        target_data_list[i]['keypoints'] = alg.kptArray2List((M_temp @ alg.homog(source_kpts))[:-1]) 

        for j in range(len(perms)): 
            perm = list(map(int,perms[j]))
            projected_kpts = (shuffleMatrix(M_temp,perm) @ alg.homog(source_kpts))[:-1] 
            target_data_lists[j][i]['keypoints'] = alg.kptArray2List(projected_kpts) 
            projected_hinv_kpts = (alg.hinv(shuffleMatrix(M_temp, perm)) @ alg.homog(source_kpts))[:-1] 
            target_hinv_lists[j][i]['keypoints'] = alg.kptArray2List(projected_hinv_kpts) 
            # try every result above and choose the one with the best score 
    
    manager.end(time()) 

    # ===========================================================
    #   save the projected kpts  
    # ===========================================================
    alg.jssave(target_data_list, os.path.join(DIR_ANSWER, f'ans_affine_naive.json'))
    for j in range(len(perms)): # 102 inv gives the answer
        alg.jssave(target_data_lists[j], os.path.join(DIR_ANSWER, f'ans_affine_{perms[j]}.json'))
        alg.jssave(target_hinv_lists[j], os.path.join(DIR_ANSWER, f'ans_affine_{perms[j]}_inv.json'))


def shuffleMatrix(M,T=(0,1,2)):
    A = M[:3,:3] 
    b = M[:3,3] 
    
    b_final = np.array([b[T[0]],b[T[1]],b[T[2]]]).reshape(3,1)
    a0 = A[:3,T[0]]; a1 = A[:3,T[1]]; a2 = A[:3,T[2]] 
    A_temp = np.hstack((a0.reshape(3,1),a1.reshape(3,1),a2.reshape(3,1))) 
    a0 = A_temp[T[0],:3]; a1 = A_temp[T[1],:3]; a2 = A_temp[T[2],:3]
    A_final = np.vstack((a0, a1, a2))

    bottom = np.array([[0,0,0,1]]) 
    M_final = np.vstack((np.hstack((A_final, b_final)), bottom))

    return M_final 

def getKptCrossSections(source_kpts, target_kpts, M_temp, source_volume, projct_volume, target_volume):
    crosses = [] 
    num_kpts = source_kpts.shape[1] 

    for i_kpt in range(num_kpts):
        source_points = source_kpts[:,i_kpt].reshape(3,1) 
        projct_points = (M_temp @ alg.homog(source_points))[:-1]
        target_points = target_kpts[:,i_kpt].reshape(3,1)  

        source_z = source_points[2,0].round().astype(int) 
        projct_z = projct_points[2,0].round().astype(int)
        target_z = target_points[2,0].round().astype(int)
        min_z = min({source_z, projct_z, target_z}) 
        max_z = max({source_z, projct_z, target_z})

        if 0:
            alg.npprint(source_points, 'source_points')
            alg.npprint(projct_points, 'projct_points') 
            alg.npprint(target_points, 'target_points') 
            alg.npprint(source_z, 'source_z') 
            alg.npprint(projct_z, 'projected_z') 
            alg.npprint(target_z, 'target_z')  
            alg.npprint(max_z, 'max_z')  
            print('shape = ', source_volume.shape)

        if min_z < 0 or max_z >= source_volume.shape[0]: continue

        source_cross = alg.vmpoints(source_volume, source_points.T)[source_z]
        projct_cross = alg.vmpoints(projct_volume, projct_points.T)[projct_z]
        target_cross = alg.vmpoints(target_volume, target_points.T, color=(0,255,0))[target_z] 

        cross = np.vstack((
            alg.imtxt(source_cross, txt=f'source:{source_points[:,0].astype(int)}'),
            alg.imtxt(projct_cross, txt=f'warped:{projct_points[:,0].astype(int)}'), 
            alg.imtxt(target_cross, txt=f'target:{target_points[:,0].astype(int)}')
            ))

        crosses.append(cross) 

        if 0:
            print(source_cross.dtype, source_cross.shape) 
            print(projct_cross.dtype, projct_cross.shape) 
            print(target_cross.dtype, target_cross.shape)
            alg.imshow(cross, f'{i_kpt}')
    
    ids = np.random.choice(len(crosses),size=5,replace=False)
    batch_crosses = [crosses[i] for i in ids]
    kpt_image = np.hstack(batch_crosses)  
    return kpt_image 

if __name__=='__main__':
    main() 
