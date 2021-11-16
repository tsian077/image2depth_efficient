import os
import glob
import time
import argparse
import efficientnet.tfkeras
# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from utils import predict, load_images, display_images, evaluate
from matplotlib import pyplot as plt
from Attention import Attention
from tensorflow import nn
from tensorflow.keras.backend import shape
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from keras import backend as K
from keras_layer_normalization import LayerNormalization
import efficientnet.keras
from pixel_shuffle import PixelShuffler
from group_norm import GroupNormalization

# import efficientnet.tfkeras
# from tensorflow.keras.models import load_model
# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
# parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
# parser.add_argument('--model', default='models/1587882651-n25344-e20-bs2-lr0.0001-Resnet_cyc_fix_nyu/model.h5', type=str, help='Trained Keras model file.')
# parser.add_argument('--model', default='/media/ee/Toshiba3T/DenseDepth/models/1588007338-n25344-e20-bs2-lr0.0001-densedepth_cyc_nyu/model.h5', type=str, help='Trained Keras model file.')
# parser.add_argument('--model', default='/media/ee/Toshiba3T/DenseDepth/models/1588222525-n25344-e20-bs2-lr0.0001-Dense-cycle-nyu-mae/model.h5', type=str, help='Trained Keras model file.') #mae
# parser.add_argument('--model', default='models/1588446405-n10138-e20-bs5-lr0.0001-Dense-gan-v2/model.h5', type=str, help='Trained Keras model file.')
# parser.add_argument('--model', default='/media/ee/Toshiba3T/DenseDepth/models/1588821665-n10138-e20-bs5-lr0.0001-Dense-gan-v2-depth-loss/model.h5', type=str, help='Trained Keras model file.')
# parser.add_argument('--model', default='models/1589421644-n10138-e20-bs5-lr0.0001-Densedepth_v2-new/model.h5', type=str, help='Trained Keras model file.')
# parser.add_argument('--model', default='models/1589617673-n10138-e20-bs5-lr0.0001-Dense_v2_gan_new_att/model.h5', type=str, help='Trained Keras model file.')# Dense-gan-attention
# parser.add_argument('--model', default='models/1589974230-n6336-e20-bs8-lr0.0001-densedepth_nyu/model.h5', type=str, help='Trained Keras model file.')
# parser.add_argument('--model', default='models/1590636045-n6336-e20-bs8-lr0.0001-Dense_at_four_state/model.h5', type=str, help='Trained Keras model file.')
# parser.add_argument('--model', default='models/1590428290-n6336-e20-bs8-lr0.0001-try_eff/model.h5', type=str, help='Trained Keras model file.')#eff
# parser.add_argument('--model', default='models/1590810527-n6336-e20-bs8-lr0.0001-dnese-gcnet-poop2pool/model.h5', type=str, help='Trained Keras model file.')
# parser.add_argument('--model', default='models/1591331055-n6336-e20-bs8-lr0.0001-b4/model.h5', type=str, help='Trained Keras model file.') #bs4
# parser.add_argument('--model', default='models/1594224573-n7242-e20-bs7-lr0.0001-b5_no_batchnorm_bs7_noisy_pixelshuffle_group_no/model.h5', type=str, help='Trained Keras model file.') #b5-pixelshuffle
# parser.add_argument('--model', default='/home/jimmy/DenseDepth/models/1596695607-n7242-e20-bs7-lr0.0001-b5_bs8_noisy_pixshuff_group_behur_sobal_color_flip_no_mirr/model.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--model', default='/home/jimmy/DenseDepth/models/1597123657-n7242-e20-bs7-lr0.0001-b5_bs8_noisy_pixshuff_group_l1_sobal_color_flip_mirr/model.h5', type=str, help='Trained Keras model file.')
args = parser.parse_args()

# model_name = 'hi-unreal'
model_name ='best'


# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D,'Attention':Attention, 'depth_loss_function': depth_loss_function,'LayerNormalization':LayerNormalization,'PixelShuffler':PixelShuffler,'GroupNormalization':GroupNormalization}

# Load model into GPU / CPU
print('Loading model...')
# model = load_model(args.model, custom_objects=custom_objects, compile=False)
model = load_model(args.model,custom_objects=custom_objects, compile=False)

# Load test data
print('Loading test data...', end='')
import numpy as np
from data import extract_zip
data = extract_zip('nyu_test.zip')
from io import BytesIO
rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
depth = np.load(BytesIO(data['eigen_test_depth.npy']))
# rgb = np.load('shop-flying-image.npy')
# depth = np.load('shop-flying-depth.npy')
# rgb = np.load('unreal_image.npy')
# depth = np.load('unreal_depth.npy')
crop = np.load(BytesIO(data['eigen_test_crop.npy']))
print('Test data loaded.\n')

start = time.time()
print('Testing...')
# print(depth)
e = evaluate(model, rgb, depth, crop, batch_size=6, model_name=model_name)

print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))


fp = open("./"+model_name+"/evaluate.txt",'w+')
fp.write("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
fp.write("\n")
fp.write("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))
fp.close
end = time.time()
print('\nTest time', end-start, 's')
