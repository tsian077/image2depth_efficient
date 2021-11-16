from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.pretrained import pspnet_50_ADE_20K
from keras_segmentation.models.pspnet import pspnet_50
import os

pretrained_model = pspnet_50_ADE_20K()

new_model = pspnet_50( n_classes=41 )

transfer_weights( pretrained_model , new_model  ) # transfer weights from pre-trained model to your model

# path = '/tmp/fine_tune_pspnet'
# os.makedirs(path)
new_model.train(
    train_images =  "nyu_seg/NYUD_v2_img/all",
    train_annotations = "nyu_seg/NYUD_v2_seq/all",
    checkpoints_path =  "tmp/pspnet_ep5_1" , epochs=5
)
out = new_model.predict_segmentation(
    inp="nyu_seg/NYUD_v2_sem_gt/test/1449-6.png",
    out_fname="tmp/pspnet_ep5.png"
)