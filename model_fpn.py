import sys

from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate,Multiply,GlobalAveragePooling2D,Reshape,BatchNormalization
from layers import BilinearUpSampling2D
from loss import depth_loss_function
import efficientnet.keras as efn 
from keras import backend as K
def create_model_eff(existing='', is_twohundred=False, is_halffeatures=True):
        
    if len(existing) == 0:
        print('Loading base model (DenseNet)..')

        # Encoder Layers
        if is_twohundred:
            print("dense201")
            base_model = applications.DenseNet201(input_shape=(None, None, 3), include_top=False)
        else:
            print('EfficientNetB6')
            base_model = efn.EfficientNetB3(input_shape=(480,640,3),include_top=False,weights='noisy-student')
            # base_model = efn.EfficientNetB4(input_shape=(480,640,3),include_top=False,weights='imagenet')
            # base_model = efn.EfficientNetB5(input_shape=(480,640,3),include_top=False,weights='imagenet')
            # base_model = efn.EfficientNetB6(input_shape=(480,640,3),include_top=False,weights='imagenet')
            # base_model = applications.DenseNet169(input_shape=(None, None, 3), include_top=False)
        from contextlib import redirect_stdout
        with open('model_b5_summary.txt', 'w') as f:

            with redirect_stdout(f): base_model.summary()
        print('Base model loaded.')

        # Starting point for decoder
        base_model_output_shape = base_model.layers[-1].output.shape

        # Layer freezing?
        for layer in base_model.layers: layer.trainable = True

        # Starting number of decoder filters
        if is_halffeatures:
            decode_filters = int(int(base_model_output_shape[-1])/2)
        else:
            decode_filters = int(base_model_output_shape[-1])

        # Define upsampling layer
        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
            up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
            up_i = BatchNormalization(axis=3)(up_i)
            # #se-block
            # se_up_i = GlobalAveragePooling2D(data_format='channels_last',name=name+'_Global')(up_i)
            # ip_shape = K.int_shape(se_up_i)
            # batchsize, channels = ip_shape
            # se_up_i = Reshape((1,1,channels))(se_up_i)
            # reduce_channel = int(channels*0.25)
            # se_up_i = Conv2D(filters=reduce_channel,kernel_size=1,strides=1,padding='same',name=name+'_se_reduce')(se_up_i)
            # se_up_i = Conv2D(filters=channels,kernel_size=1,strides=1,padding='same',name=name+'_se_expand')(se_up_i)
            # up_i = Multiply()([up_i,se_up_i])
            #se-block
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = BatchNormalization(axis=3)(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = BatchNormalization(axis=3)(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.output)

        #Eff
        

        #Dense
        # decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
        # decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
        # decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
        # decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
        # if False: decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')
        #Effective
        
        
        decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='block4a_dwconv')
        decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='block3a_dwconv')
        decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='block2a_dwconv')
        decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='stem_conv')
        # decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='')
        if False: decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')

        # Extract depths (final layer)
        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

        # Create the model
        model = Model(inputs=base_model.input, outputs=conv3)
    else:
        # Load model from file
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects)
        print('\nExisting model loaded.\n')

    print('Model created.')
    
    return model