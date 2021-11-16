import sys

from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate,Dropout
from keras_layer_normalization import LayerNormalization
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from Attention import Attention
# from gc_block import gc_block
import re
from non_local import non_local_block
def dropout_layer_factory():
    print('dropout',type(Dropout(rate=0.2, name='dropout')))
    return Dropout(rate=0.2, name='dropout')

def non_local_layer_factory():
    return non_local_block(compression=2,mode='embedded')
def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')
            print('insert',type(insert_layer_factory))
            # new_layer = insert_layer_factory()
            new_layer = insert_layer_factory
            if insert_layer_name:
                print('1')
                new_layer.name = insert_layer_name
            else:
                print('2')
                print(layer.name)
                new_layer_name = layer.name+'_non_local'
            # x = new_layer(x)
            #non-local
            # x = non_local_block(x,compression=2,mode='dot')
            #gcnet
            x = gc_block(x,mode='gc',compression=2)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer_name,
                                                            layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)

def create_model(existing='', is_twohundred=False, is_halffeatures=True):
        
    if len(existing) == 0:
        print('Loading base model (DenseNet)..')

      


        #Generator
        # Encoder Layers
        if is_twohundred:
            base_model = applications.DenseNet201(input_shape=(None, None, 3), include_top=False)
        else:
            print('load pretrained DenseNet169')
            base_model = applications.DenseNet169(input_shape=(480, 640, 3), include_top=False,weights="imagenet")
            # base_model = applications.DenseNet169(input_shape=(None, None, 3), include_top=False,weights=None)
            # base_model = applications.ResNet50(input_shape=(480, 640, 3), include_top=False)

        print('Base model loaded.')
       
        # Starting point for decoder
        base_model_output_shape = base_model.layers[-1].output.shape
        # print(base_model_output_shape)
        # Starting number of decoder filters
        if is_halffeatures:
            decode_filters = int(int(base_model_output_shape[-1])/2)
        else:
            decode_filters = int(base_model_output_shape[-1])

        
        base_model = insert_layer_nonseq(base_model,'pool2_pool',non_local_layer_factory)
        base_model.save('temp.h5')
        custom_objects = {'LayerNormalization':LayerNormalization}
        base_model = load_model('temp.h5', custom_objects=custom_objects)
        print(base_model.summary())
        
        # print(base_model)
        # print(pool2_pool_output)
        # print(type(pool2_pool_output))
        # base_model = insert_layer_nonseq(base_model,'pool2_pool',non_local_block(pool2_pool_output,mode='embedded',compression=2))
        
        
        # Layer freezing?
        for layer in base_model.layers: layer.trainable = True
       
        
        

       

       

        # Define upsampling layer
        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
            up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.get_layer('relu').output)
        # non_local = gc_block(decoder,mode='gc',compression=2)
        # non_local = non_local_block(decoder,mode='embedded',compression=2)
        decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
        if False: decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')
        
        #resnet
        # decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='res4a_branch2a')
        # decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='res3a_branch2a')
        # decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='res2a_branch2a')
        # decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='activation_1')
        # if False: decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')

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