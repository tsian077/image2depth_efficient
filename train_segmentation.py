from keras_segmentation.models.unet import resnet50_unet
# from keras_segmentation.models.segnet import mobilenet_segnet

model = resnet50_unet(n_classes=41 ,  input_height=480, input_width=640)

# from contextlib import redirect_stdout
# with open('model_summary.txt', 'w') as f:


#     with redirect_stdout(f): model.summary()
model.train(
    train_images =  "nyu_seg/NYUD_v2_img/all",
    train_annotations = "nyu_seg/NYUD_v2_seq/all",
    checkpoints_path = "tmp/resnet50_ep500_1" , epochs=500
)

out = model.predict_segmentation(
    inp="nyu_seg/NYUD_v2_sem_gt/test/1449-6.png",
    out_fname="tmp/resnet50_ep500.png"
)
