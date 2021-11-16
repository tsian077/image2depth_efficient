import open3d as o3d
import matplotlib.pyplot as plt
import os
import cv2


from PIL import Image




home = os.getcwd()


size = 3
path = os.path.join(home,'effiectnet4')

# gt_path = os.path.join(path,'depth_ground_np')
# pd_path = os.path.join(path,'depth_np')


depth_path = os.path.join(path,'origin_depth')
image_path = os.path.join(path,'image')

for i in range(size):
    
    im = Image.open(image_path+'/image_{i}.png'.format(i=i))
    rgb_im = im.convert('LA')
    rgb_im.save(image_path+'/image_{i}.png'.format(i=i))
    

# error_map_path = os.path.join(path,'')

# if not os.path.exists(error_map_path):
#     os.mkdir(error_map_path)



for i in range(size):

    

    # depth = cv2.imread(depth_path+'/depth_{i}.png'.format(i=i+1))
    # image = cv2.imread(image_path+'/image_{i}.png'.format(i=i))
    # print(depth.shape)
    # print(image.shape)
    # plt.subplot(1, 2, 1)
    # plt.title('SUN grayscale image')
    # plt.imshow(depth)
    # plt.subplot(1, 2, 2)
    # plt.title('SUN depth image')
    # plt.imshow(image)
    # plt.show()

    color_raw = o3d.io.read_image(image_path+'/image_{i}.png'.format(i=i))
    depth_raw = o3d.io.read_image(depth_path+'/depth_{i}.png'.format(i=i+1))
    print(color_raw)
    print(depth_raw)

    rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(color_raw, depth_raw)

    
    plt.subplot(1, 2, 1)
    plt.title('SUN grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('SUN depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()
    
    
    break
    
   


