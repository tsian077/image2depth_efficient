import os
import numpy as np
import scipy.misc
import cv2
import matplotlib.pyplot as plt
import skimage

home = os.getcwd()

print(home)
size = 600
path = os.path.join(home,'efficientb5-pixelshuffle-groupnor-bs7')
path2 = os.path.join(home,'efficientb5-pixelshuffle-groupnor-bs6')
# gt_path = os.path.join(path,'depth_ground_np')
# pd_path = os.path.join(path,'depth_np')


gt_path = os.path.join(path,'depth_ground')
# gt_path = os.path.join(home,'dense-gcnet','depth')
pd_path = os.path.join(path,'depth')
pd2_path = os.path.join(path2,'depth')

image_path = os.path.join(path,'image2')



error_map_path = os.path.join(path,'error_map')

if not os.path.exists(error_map_path):
    os.mkdir(error_map_path)



for i in range(size):

    # gt = np.load(gt_path+'/{i}.npy'.format(i=i))
    # pd = np.load(pd_path+'/{i}.npy'.format(i=i))
    gt = cv2.imread(gt_path+'/depth_{i}.png'.format(i=i))
    pd = cv2.imread(pd_path+'/depth_{i}.png'.format(i=i))
    pd2 = cv2.imread(pd2_path+'/depth_{i}.png'.format(i=i))
    image = cv2.imread(image_path+'/image_{i}.png'.format(i=i+1))
    
    a = np.abs(gt-pd)
    b = np.abs(gt-pd2) 
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,3,1)
    ax1.set_title('ground truth')
    gt = cv2.applyColorMap(gt,cv2.COLORMAP_JET)
    ax1.imshow(gt)
    ax2 = fig.add_subplot(2,3,2)
    ax2.set_title('predict bs7')
    pd = cv2.applyColorMap(pd,cv2.COLORMAP_JET)
    ax2.imshow(pd)
    ax3 = fig.add_subplot(2,3,3)
    ax3.set_title('predict bs6')
    pd2 = cv2.applyColorMap(pd2,cv2.COLORMAP_JET)
    ax3.imshow(pd2)
    ax4 = fig.add_subplot(2,3,4)
    ax4.set_title('input image')
    ax4.imshow(image)

    
    a = cv2.applyColorMap(a, cv2.COLORMAP_JET)
    ax5 = fig.add_subplot(2,3,5)
    
    ax5.set_title('error map1')
    ax5.imshow(a)

    b = cv2.applyColorMap(b, cv2.COLORMAP_JET)
    ax5 = fig.add_subplot(2,3,6)
    
    ax5.set_title('error map2')
    ax5.imshow(b)
    
    plt.show()

    
    # break


    # print(gt)

    

    # print('=============')
    # print(pd)
    
    # print('--------------')
    # a= np.abs(gt-pd)
   
    # print(a)
    
    # print(i)
    # a = cv2.applyColorMap(a, cv2.COLORMAP_JET)
    # fig = plt.figure()
    # ii = plt.imshow(a)
    # fig.colorbar(ii)
    # plt.show()
    
    # break
    # 
    # fig = plt.figure(figsize = (W,H))
    # pos = fig.add_axes([0.93,0.1,0.02,0.35]) # Set colorbar position in fig
    
    # fig.colorbar(a, cax=pos) # Create the colorbar
    # plt.colorbar()
    # plt.imsave("out.png", a, cmap = 'hot')
    # plt.savefig("out.png", a, cmap = 'hot')
    # break
    
    # cv2.imwrite(error_map_path+'/error_map_{i}.jpg'.format(i=i),a)
    

    
    
    # scipy.misc.imsave(error_map_path+'/error_map_{i}.png'.format(i=i),a)
    
    
   


