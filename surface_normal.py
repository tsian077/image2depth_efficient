import os
import numpy as np
import scipy.misc
import cv2
import matplotlib.pyplot as plt


home = os.getcwd()

print(home)
size = 3
path = os.path.join(home,'effnet')

# gt_path = os.path.join(path,'depth_ground_np')
# pd_path = os.path.join(path,'depth_np')


gt_path = os.path.join(path,'depth_ground_np')
pd_path = os.path.join(path,'depth_np')


surface_normal_path = os.path.join(path,'surface_normal')

if not os.path.exists(surface_normal_path):
    os.mkdir(surface_normal_path)
def compute_normals(y):
    zx = cv2.Sobel(y,cv2.CV_64F,1,0,ksize=5)
    zy = cv2.Sobel(y,cv2.CV_64F,0,1,ksize=5)

    normal = np.dstack((-zx,-zy,np.ones_like(y)))

    n = np.linalg.norm(normal,axis=2)

    normal[:,:,0] /= n
    normal[:,:,1] /= n
    normal[:,:,2] /= n
    normal +=1 
    normal /=2
    normal *=255
    return normal


for i in range(size):

    
    gt = np.load(gt_path+'/{i}.npy'.format(i=i))
    pd = np.load(pd_path+'/{i}.npy'.format(i=i))
    print(gt)
    print('------')
    print(pd)
    # gt = cv2.imread(gt_path+'/{i}.png'.format(i=i))
    # pd = cv2.imread(pd_path+'/depth_{i}.png'.format(i=i))
    
    gt_sn=compute_normals(pd)
    pd_sn=compute_normals(pd)
    # print(gt_sn.shape)
    cv2.imwrite(surface_normal_path+'/gt_{i}.png'.format(i=i),gt_sn[:,:,::-1])
    cv2.imwrite(surface_normal_path+'/pd_{i}.png'.format(i=i),pd_sn[:,:,::-1])

    # print(gt)
    # print(pd)
    


 
    
   


