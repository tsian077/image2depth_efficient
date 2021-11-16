# from data import get_nyu_train_test_data

# train_generator, test_generator = get_nyu_train_test_data( 1 )

# import numpy as np 
# from data import extract_zip
# data = extract_zip('nyu_test.zip')
# from io import BytesIO
# rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
# depth = np.load(BytesIO(data['eigen_test_depth.npy']))
# crop = np.load(BytesIO(data['eigen_test_crop.npy']))
# print('Test data loaded.\n')




import numpy as np
from PIL import Image
depth = np.load('eigen_test_depth.npy')
print('depth_ground',depth[0])
print(depth[0].max())
# depth = depth
print(np.amax(depth))
# print('======================')
# unreal_depth = np.load('/home/jimmy/DenseDepth/unreal/depth/dapth_0.npy')
# print(unreal_depth)
# print(np.amax(unreal_depth))
# print("----------------------")

# unreal_y = np.asarray(Image.open( '/home/jimmy/DenseDepth/unreal/depth_image/depth_image_0.png' )).reshape(480,640,3)/255*1000
# unreal_y=np.load('/home/jimmy/DenseDepth/unreal/depth/dapth_0.npy').reshape(480,640,1)
# y = np.asarray(Image.open( '/home/jimmy/DenseDepth/data/nyu2_train/living_room_0038_out/37.png' )).reshape(480,640,1)/255*1000
# # y = np.clip(np.asarray(Image.open( '/home/jimmy/DenseDepth/data/nyu2_train/living_room_0038_out/37.png' )).reshape(480,640,1)/255*1000,0,1000)
# print(unreal_y)
# print(unreal_y.max())
# print("=-------------=")
# print(y.max())
# from PIL import Image

# # y = np.asarray(Image.open('/home/jimmy/DenseDepth/data/nyu2_test/00000_depth.png'), dtype=np.float32).reshape(480,640,1).copy().astype(float) /10.0

# y = np.asarray(Image.open( '/home/jimmy/DenseDepth/hi-unreal/depth/depth_103.png' ))
# y = np.load('/home/jimmy/DenseDepth/unreal/depth/dapth_103.npy')
# print(y.max())


# print(y)
# print(y.shape)

# import numpy as np
