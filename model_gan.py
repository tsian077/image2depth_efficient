import sys

from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate,Conv2DTranspose,BatchNormalization,Reshape
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from keras.optimizers import Adam
from Attention import Attention
from non_local import non_local_block
# from Reshapesize import ResizeImages



# build_generator
def build_generator(existing='', is_twohundred=False, is_halffeatures=True):
        
    if len(existing) == 0:
        print('Loading base model (ResNet)..')

    
    

        # Encoder Layers
        if is_twohundred:
            base_model = applications.DenseNet201(input_shape=(None, None, 3), include_top=False)
        else:
            
            print('Loading Dense169') 
            # base_model = applications.ResNet50(input_shape=(480, 640, 3), include_top=False)
            base_model = applications.DenseNet169(input_shape=(None, None, 3), include_top=False,weights="imagenet")

        print('Base model loaded.')

        # Starting point for decoder
        base_model_output_shape = base_model.layers[-1].output.shape
        print('last-layer',base_model_output_shape)
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
            print(up_i)
            print(base_model.get_layer(concat_with).output,'concat_with')
            up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i
        
        

        
        #Resnet Decoder Layers
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.output)

        #res
        # decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='res4a_branch2a')
        # decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='res3a_branch2a')
        # decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='res2a_branch2a')
        # decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='activation_1')
        # if False: decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')

        #dense
        decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
        att = Attention(int(decode_filters/2))(decoder)
        decoder = upproject(att, int(decode_filters/4), 'up2', concat_with='pool2_pool')
        att = Attention(int(decode_filters/4))(decoder)
        decoder = upproject(att, int(decode_filters/8), 'up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
        if False: decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')

        # Extract depths (final layer)
        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

        print('last',conv3)

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

def build_discriminator():

    df = 64
    img_rows = 480
    img_cols = 640
    channels = 3 
    img_shape = (img_rows,img_cols,channels)
    resimg_shape = (240,320,channels)
    depth_shape= (240,320,1)
    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img_A = Input(shape=depth_shape)

    # img_B = Input(shape=img_shape)
    # img_B = ResizeImages(output_dim=(240,320))(img_B)
    img_C = Input(shape=resimg_shape)
    img =  Concatenate(axis=-1)([img_A, img_C])
    d1 = d_layer(img, df, normalization=False)
    
    d2 = d_layer(d1, df*2)
    
    d3 = d_layer(d2, df*4)
    d4 = d_layer(d3, df*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model([img_A,img_C], validity)





def create_model_gan(existing='',is_twohundred=False,is_halffeatures=True):
    img_rows = 480
    img_cols = 640
    channels = 3 
    img_shape = (img_rows,img_cols,channels)
    resimg_shape = (240,320,channels)
    Depth_shape = (240,320,1)

    patch = int(img_rows / 2**4)
    disc_patch = (patch,patch,1)

    optimizer = Adam(0.0001, amsgrad=True)

    #Discrimator
    Discriminator = build_discriminator()
    
    Discriminator.trainable=False

    Discriminator.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])

    #Generator
    Generator = build_generator(existing='',is_twohundred=False,is_halffeatures=True)
    # for layer in self.discriminator.layers:
    #     print (layer.get_weights())
    #img_B:RGB img_A:Depth
    # img_A = Input(shape=Depth_shape)
    img_B = Input(shape=img_shape)
    img_A = Input(shape=Depth_shape)
    img_C = Input(shape=resimg_shape)
    fake_A = Generator(img_B)

    Discriminator.trainable = True
    
    valid = Discriminator([img_A,img_C])

    
    fake  = Discriminator([fake_A,img_C])
    

    combined = Model(inputs=[img_B,img_A,img_C],output=[fake,valid,fake_A])
    combined.compile(loss=['mse','mse',depth_loss_function],
                          loss_weights=[1,1,1],
                          optimizer=optimizer)
    # combined.compile(loss=['mse','mse','mse'],
    #                       loss_weights=[1,1,1],
    #                       optimizer=optimizer)
    # combined = Model(inputs=img_B,output=fake_A)
    # combined.compile(loss=depth_loss_function,optimizer=optimizer)

    return combined

    

