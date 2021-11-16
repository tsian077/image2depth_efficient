import keras.backend as K
import tensorflow as tf

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
    
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    #huber
    # l = tf.keras.losses.Huber()
    # l_huber = l(y_pred, y_true)

    #behur
    # absdiff = K.abs(y_pred-y_true)
    # C = 0.2*K.max(absdiff)
    # l_behur = K.mean(tf.where(absdiff<C,absdiff,(absdiff*absdiff+C*C)/(2*C)))
    # # https://github.com/abduallahmohamed/reversehuberloss/blob/master/rhuloss.py

    

    # Edges
    # dy_true, dx_true = tf.image.image_gradients(y_true)
    # dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    # l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    #Sobal
    sobal_true = tf.image.sobel_edges(y_true)
    sobal_pred = tf.image.sobel_edges(y_pred)
    l_sobal = K.mean(K.abs(sobal_pred - sobal_true),axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta
    # return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * l_behur) #huber
    # return (w1 * l_ssim) + (w2 * K.mean(l_sobal)) + (w3 * l_behur) #huber

    return (w1 * l_ssim) + (w2 * K.mean(l_sobal)) + (w3 * K.mean(l_depth)) #sobal

  

    # return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth)) #origin
    # return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * l_huber) #huber
    # return (w1 * l_ssim)  + (w3 * K.mean(l_depth)) #not good


# a=np.random.rand(5,5,5,5)
# b=np.random.rand(5,5,5,5)
# c = np.mean(np.abs(a-b),axis=-1)
# d=np.mean(c)

# e=np.mean(np.abs(a-b))
# 0.3250013149947966 == 0.3250013149947967
# e==d