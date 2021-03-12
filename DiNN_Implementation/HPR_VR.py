# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:32:21 2019

@author: Haoti
"""

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time

##### SECTION TO RUN WITH GPU #####

# Choose GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU ID to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0";

Config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
Config.gpu_options.allow_growth=True

# Load data
# import Cartesian Map contours
# CM size 0.25mm

rectangle_x = 0.01
rectangle_y = 0.01

data_set = 1000
epsilon  = 1e-10

n_x = 79
n_y = 79
grid_dist_x = rectangle_x / n_x
grid_dist_y = rectangle_y / n_y

m_x = int(n_x + 1)
m_y = int(n_y + 1)

data_cart_new   = np.zeros((data_set, m_x, m_y))
one_cart_new    = np.zeros((data_set, m_x, m_y))
data_stress_new = np.zeros((data_set, m_x, m_y))
one_stress_new  = np.zeros((data_set, m_x, m_y))
data_property   = np.zeros((data_set, m_x, m_y))

start_1 = time()

for k in range(data_set):
    
    idx = str(k+1)
    
    txt_cart   = 'C:\\Temp_Abaqus\\micro_meter_model\\fix_hole_hollow_ring_new\\Add_SDF_Cart_' + idx + '.dat'
    txt_stress = 'C:\\Temp_Abaqus\\micro_meter_model\\fix_hole_hollow_ring_new\\New_Composite_uniform_Stress_Cart_' + idx + '.dat'
    one_cart      = 'C:\\Temp_Abaqus\\micro_meter_model\\fix_hole_hollow_ring_new\\One_SDF_Cart_' + idx + '.dat'
    one_stress    = 'C:\\Temp_Abaqus\\micro_meter_model\\fix_hole_hollow_ring_new\\One_Stress_Cart_' + idx + '.dat'

    data_cart   = np.loadtxt(txt_cart)
    data_stress = np.loadtxt(txt_stress)
    one_cart    = np.loadtxt(one_cart)
    one_stress  = np.loadtxt(one_stress)

    [m,n] = data_cart.shape

    for i in range(m): 
        x = int(round(data_cart[i][0] / grid_dist_x))
        y = int(round(data_cart[i][1] / grid_dist_y))
        data_cart_new[k][y][x] = data_cart[i][2]
        one_cart_new[k][y][x]  = one_cart[i][2]
        one_stress_new[k][y][x]  = one_stress[i][2]
        data_stress_new[k][y][x] = data_stress[i][2]
        data_property[k][y][x]   = data_stress[i][3]

# Data preprocessing     
X_train, X_test, Y_train, Y_test, P_train, P_test, One_cart_train, One_cart_test, One_stress_train, One_stress_test = train_test_split(data_cart_new,data_stress_new,data_property,one_cart_new,  one_stress_new,  test_size=0.2, random_state=47)
X_test,  X_cv,   Y_test,  Y_cv,   P_test,  P_cv,   One_cart_test,  One_cart_cv,   One_stress_test,  One_stress_cv   = train_test_split(X_test,       Y_test,         P_test,       One_cart_test, One_stress_test, test_size=0.5, random_state=47)

One_cart_train   = tf.reshape(One_cart_train,  [-1, m_x, m_y, 1])
One_stress_train = tf.reshape(One_stress_train, [-1, m_x, m_y, 1])
One_cart_test    = tf.reshape(One_cart_test,   [-1, m_x, m_y, 1])
One_stress_test  = tf.reshape(One_stress_test, [-1, m_x, m_y, 1])
One_cart_cv      = tf.reshape(One_cart_cv,     [-1, m_x, m_y, 1])
One_stress_cv    = tf.reshape(One_stress_cv,   [-1, m_x, m_y, 1])

input_train      = tf.reshape(X_train, [-1, m_x, m_y, 1])
output_train     = tf.reshape(Y_train, [-1, m_x, m_y, 1])
#output_train_new = tf.math.divide(output_train, K_train+epsilon)

sdf_ave  = tf.reduce_mean(input_train, 0)
sdf_ave  = tf.reshape(sdf_ave, [-1, m_x, m_y, 1])
stress_ave = tf.reduce_mean(output_train, 0)
stress_ave = tf.reshape(stress_ave, [-1, m_x, m_y, 1])

input_train_new  = input_train - sdf_ave
#input_train_new  = tf.math.multiply(input_train_new, One_cart_train)
output_train_new = output_train - stress_ave
[s1,s2,s3,s4]    = input_train.shape
stress_ave_train = tf.keras.backend.repeat_elements(stress_ave, rep=s1, axis=0)  

input_cv  = tf.reshape(X_cv, [-1, m_x, m_y, 1])
output_cv = tf.reshape(Y_cv, [-1, m_x, m_y, 1])
#output_cv = tf.math.multiply(output_cv, Ref_cv)
input_cv_new  = input_cv - sdf_ave
#input_cv_new  = tf.math.multiply(input_cv_new, One_cart_cv)
[c1,c2,c3,c4] = input_cv.shape
stress_ave_cv = tf.keras.backend.repeat_elements(stress_ave, rep=c1, axis=0)


input_test  = tf.reshape(X_test, [-1, m_x, m_y, 1])
output_test = tf.reshape(Y_test, [-1, m_x, m_y, 1])
#output_test = tf.math.multiply(output_test, Ref_test)
input_test_new    = input_test - sdf_ave
#input_test_new    = tf.math.multiply(input_test_new, One_cart_test)
[te1,te2,te3,te4] = input_test.shape
stress_ave_test   = tf.keras.backend.repeat_elements(stress_ave, rep=te1, axis=0) 

############################ Normalizing input sdf ############################

max_sdf     = np.max(input_train_new)
max_stress  = np.max(output_train_new)
min_sdf     = np.min(input_train_new)
min_stress  = np.min(output_train_new)

input_train_new = (input_train_new - min_sdf) / (max_sdf - min_sdf) # min-max norm
input_train_new = tf.math.multiply(input_train_new, One_cart_train)
input_cv_new    = (input_cv_new - min_sdf)    / (max_sdf - min_sdf)
input_cv_new    = tf.math.multiply(input_cv_new, One_cart_cv)
input_test_new  = (input_test_new - min_sdf)  / (max_sdf - min_sdf)
input_test_new  = tf.math.multiply(input_test_new, One_cart_test)

t1 = time() - start_1
start_2 = time()

########################### Defining Neural Network ###########################

def conv_relu_block(x,filt,names):
    
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[2,2], strides=2,
                               padding='same', activation='linear', 
                               use_bias=True,name=names)(x)
    y = tf.keras.layers.ReLU()(y)

    y = tf.keras.layers.BatchNormalization()(y)
    
    return y

def se_block(x,filt,ratio=16):
    
    init = x
    se_shape = (1, 1, filt)
    
    se = tf.keras.layers.GlobalAveragePooling2D()(init)
    se = tf.keras.layers.Reshape(se_shape)(se)
    se = tf.keras.layers.Dense(filt // ratio, activation='relu', 
                               kernel_initializer='he_normal', 
                               use_bias=False)(se)
    se = tf.keras.layers.Dense(filt, activation='sigmoid', 
                               kernel_initializer='he_normal', 
                               use_bias=False)(se)
    se = tf.keras.layers.multiply([init, se])
    
    return se

def resnet_block(x,filt):

    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[3,3], 
                               padding='same', activation='linear', 
                               use_bias=True)(x)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)
    
    y = tf.keras.layers.Conv2D(filters=filt, kernel_size=[3,3], 
                               padding='same', activation='linear',
                               use_bias=True)(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.BatchNormalization()(y)

    y = se_block(y,filt)
     
    y = tf.keras.layers.Add()([y,x])
    
    return y

def deconv_norm_linear(x,filt,kernel,stride,names):
    
    y = tf.keras.layers.Conv2DTranspose(filters=filt,kernel_size=kernel,
        strides=stride,padding='same',activation='linear', use_bias=True,
        name=names)(x)
    
    y = tf.keras.layers.Activation(activation='linear')(y)
    
    y = tf.keras.layers.BatchNormalization()(y)

    return y

def deconv_norm_sigmoid(x,filt,kernel,stride,names):
    
    y = tf.keras.layers.Conv2DTranspose(filters=filt, kernel_size=kernel,
        strides=stride, padding='same', activation='linear', use_bias=True,
        name=names)(x)
    
    y = tf.keras.layers.Activation(activation='sigmoid')(y)
    
    y = tf.keras.layers.BatchNormalization()(y)

    return y

def deconv_norm_relu(x,filt,kernel,stride,names):
    
    y = tf.keras.layers.Conv2DTranspose(filters=filt, kernel_size=kernel,
        strides=stride, padding='same', activation='linear', use_bias=True,
        name=names)(x)
    
    y = tf.keras.layers.ReLU()(y)
    
    y = tf.keras.layers.BatchNormalization()(y)

    return y

def deconv_block(x,filt,kernel,stride,names):
    
    y = tf.keras.layers.Conv2DTranspose(filters=filt, kernel_size=kernel,
        strides=stride, padding='same', activation='linear', use_bias=True,
        name=names)(x)
    
    y = tf.keras.layers.BatchNormalization()(y)

    return y

def dense_block(x,filt,names):
    
    y = tf.keras.layers.Dense(filt, activation='linear',
                              kernel_initializer='he_normal', use_bias=False,
                              name=names)(x)
    
    return y

######################### Construct Neural Network ############################

input_layer_1 = tf.keras.Input(shape=(m_x, m_y, 1))
input_layer_1 = tf.cast(input_layer_1, tf.float32)

input_layer_2 = tf.keras.Input(shape=(m_x, m_y, 1))
input_layer_2 = tf.cast(input_layer_2, tf.float32)

input_layer_3 = tf.keras.Input(shape=(m_x, m_y, 1))
input_layer_3 = tf.cast(input_layer_3, tf.float32)

conv_1 = conv_relu_block(input_layer_1, 32, 'conv1')
se_1   = se_block(conv_1, 32)
conv_2 = conv_relu_block(se_1, 64, 'conv2')
se_2   = se_block(conv_2, 64)
conv_3 = conv_relu_block(se_2, 128, 'conv3')
se_3   = se_block(conv_3, 128)

resnet_1 = resnet_block(se_3, 128)
resnet_2 = resnet_block(resnet_1, 128)
resnet_3 = resnet_block(resnet_2, 128)
resnet_4 = resnet_block(resnet_3, 128)
resnet_5 = resnet_block(resnet_4, 128)

deconv_0 = deconv_norm_linear(resnet_5, 128, [2,2], (2,2), 'deconv0')
deconv_1 = deconv_norm_linear(deconv_0,  64, [2,2], (2,2), 'deconv1')
deconv_2 = deconv_norm_linear(deconv_1,  32, [2,2], (2,2), 'deconv2')
deconv_3 = deconv_norm_linear(deconv_2,   1, [2,2], (1,1), 'deconv3')

deconv_3 = deconv_3 * (max_stress - min_stress) + min_stress

dense = deconv_3 + input_layer_2

deconv_4 = deconv_norm_linear(dense, 1, [2,2], (1,1), 'deconv4')

deconv_4 = tf.keras.layers.ReLU()(deconv_4)

deconv_4 = tf.math.multiply(deconv_4, input_layer_3)

output_layer = deconv_4

model = tf.keras.models.Model(inputs=[input_layer_1,input_layer_2,input_layer_3], outputs=output_layer)

model.summary()

######################## Training the model ########################

sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.6, nesterov=True)
model.compile(optimizer=sgd, loss=tf.keras.losses.mean_squared_error, metrics=['accuracy'])

epoch = 80
history = model.fit([input_train_new, stress_ave_train, One_stress_train], output_train, batch_size=256, epochs=epoch, 
                    steps_per_epoch=40, validation_data=([input_cv_new, stress_ave_cv, One_stress_cv], output_cv))

# new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0])
# new_cols = ((cols - 1) * strides[1] + kernel_size[1] - 2 * padding[1] + output_padding[1])
# Evaluate the model on test set

predict = model.predict([input_test_new, stress_ave_test, One_stress_test])

score = model.evaluate([input_test_new, stress_ave_test, One_stress_test], output_test, verbose=1)
print('\n', 'Test accuracy', score)

t2 = time() - start_2 

### Generating history plots of training ###

# Summarize history for accuracy
fig_acc = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy in training')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig('training_accuracy.png')

# Summarize history for loss
fig_loss = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_loss.savefig('training_loss.png')

### Plot and generate graphs for test samples ###
sdf_ave_plot = sdf_ave[0,:,:,0]

fig0_sdf_ave = plt.figure()
plt.title('SDF average')
plt.imshow(sdf_ave_plot)
plt.colorbar()
plt.grid(True)
plt.show()
fig0_sdf_ave.savefig('sdf_ave.png')

stress_ave_plot = stress_ave[0,:,:,0]

fig0_stress_ave = plt.figure()
plt.title('stress average')
plt.imshow(stress_ave_plot)
plt.colorbar()
plt.grid(True)
plt.show()
fig0_stress_ave.savefig('stress_ave.png')

# The first dataset
X_test_1 = input_test_new[0, :, :, 0]
Y_test_1 = output_test[0, :, :, 0]

fig1_sdf = plt.figure()
plt.title('SDF contour')
plt.imshow(X_test_1)
plt.colorbar()
plt.grid(True)
plt.show()
fig1_sdf.savefig('SDF_1.png')

fig1_test = plt.figure()
plt.title('Stress contour')
plt.imshow(Y_test_1)
plt.colorbar()
plt.grid(True)
plt.show()
fig1_test.savefig('Test_1.png')

predict_1 = predict[0, :, :, 0]
fig1_pred=plt.figure()
plt.title('Trained stress contour')
plt.imshow(predict_1)
plt.colorbar()
plt.grid(True)
plt.show()
fig1_pred.savefig('Predict_1.png')

# The second dataset
X_test_2 = input_test_new[3, :, :, 0]
Y_test_2 = output_test[3, :, :, 0]

fig2_sdf = plt.figure()
plt.title('SDF contour')
plt.imshow(X_test_2)
plt.colorbar()
plt.grid(True)
plt.show()
fig2_sdf.savefig('SDF_2.png')

fig2_test = plt.figure()
plt.title('Stress contour')
plt.imshow(Y_test_2)
plt.colorbar()
plt.grid(True)
plt.show()
fig2_test.savefig('Test_2.png')

predict_2 = predict[3,:,:,0]
fig2_pred=plt.figure()
plt.title('Trained stress contour')
plt.imshow(predict_2)
plt.colorbar()
plt.grid(True)
plt.show()
fig2_pred.savefig('Predict_2.png')

# The third dataset
X_test_3 = input_test_new[8, :, :, 0]
Y_test_3 = output_test[8, :, :, 0]

fig3_sdf = plt.figure()
plt.title('SDF contour')
plt.imshow(X_test_3)
plt.colorbar()
plt.grid(True)
plt.show()
fig3_sdf.savefig('SDF_3.png')

fig3_test=plt.figure()
plt.title('Stress contour')
plt.imshow(Y_test_3)
plt.colorbar()
plt.grid(True)
plt.show()
fig3_test.savefig('Test_3.png')

predict_3 = predict[8, :, :, 0]
fig3_pred=plt.figure()
plt.title('Trained stress contour')
plt.imshow(predict_3)
plt.colorbar()
plt.grid(True)
plt.show()
fig3_pred.savefig('Predict_3.png')

# Plot out what each layer is doing
## first plot the original one

input_image = input_train[0, :, :, 0]

plt.figure()
plt.title('input contour')
plt.imshow(input_image)
plt.colorbar()
plt.grid(True)
plt.show()

# Second plot what each layer is doing 
#layer_names = ['conv1','pool1','conv2','pool2','conv3','dense1','dense2',
#              'dense3','dense4','deconv1','deconv2']

layer_names = ['conv1','conv2','conv3','deconv1','deconv2','deconv3']

for layer_name in layer_names:
   
    inter_layer_model = tf.keras.models.Model(inputs=model.input,
                                          outputs=model.get_layer(layer_name).output)    
    inter_output = inter_layer_model.predict([input_test_new, stress_ave_test,One_stress_test])    
    ran = inter_output.shape[3]
    
    for j in range(ran):
        plot_inter_output = inter_output[0,:,:,j]
        layer_name_new = layer_name + str(j)
        
        fig = plt.figure()
        plt.title(layer_name_new)
        plt.imshow(plot_inter_output)
        plt.colorbar()
        plt.grid(True) 
        
        name_layer = layer_name_new + '.png'
        fig.savefig(name_layer)
    
# Third plot the real stress contour
output_image = output_train[0,:,:,0]

fig2 = plt.figure()
plt.title('stress contour')
plt.imshow(output_image)
plt.colorbar()
plt.grid(True)
plt.show()

fig2.savefig('real_stress.png')

# Calculate max stress difference in test set

predict_output = predict[:, :, :, 0]
[p1,p2,p3] = predict_output.shape

max_real_fiber  = np.zeros(p1)
max_real_matrix = np.zeros(p1)
 
min_real_fiber  = np.zeros(p1)
min_real_matrix = np.zeros(p1)

max_pred_fiber  = np.zeros(p1)
max_pred_matrix = np.zeros(p1)

min_pred_fiber  = np.zeros(p1)
min_pred_matrix = np.zeros(p1)

error_max_fiber  = np.zeros(p1)
error_max_matrix = np.zeros(p1)

error_min_fiber  = np.zeros(p1)
error_min_matrix = np.zeros(p1)

error_rate_max_fiber  = np.zeros(p1)
error_rate_max_matrix = np.zeros(p1)


for ip in range(p1):
    
    pos1 = 0
    pos2 = 0
    
    for j in range (m_x):
        for k in range (m_y):
            
            if P_test[ip][j][k] == 1: 
            #if tf.math.not_equal(X_test[ip][j][k], 1) and tf.math.not_equal(X_test[ip][j][k], 0):
                pos1 = pos1 + 1
                
            elif P_test[ip][j][k] == 2:
            #elif X_test[ip][j][k] == 1: 
                pos2 = pos2 + 1
    
    fiber_test_stress  = None
    fiber_pred_stress  = None
    matrix_test_stress = None
    matrix_pred_stress = None
            
    fiber_test_stress  = np.zeros(pos1)
    fiber_pred_stress  = np.zeros(pos1)
    matrix_test_stress = np.zeros(pos2)    
    matrix_pred_stress = np.zeros(pos2)

    vector_idx_1 = 0
    vector_idx_2 = 0

    for j in range(m_x):
        for k in range(m_y):
            
            if P_test[ip][j][k] == 1:
            #if tf.math.not_equal(X_test[ip][j][k], 1) and tf.math.not_equal(X_test[ip][j][k], 0):
                
                fiber_test_stress[vector_idx_1] = Y_test[ip,j,k]
                fiber_pred_stress[vector_idx_1] = predict[ip,j,k]
                vector_idx_1 = vector_idx_1 + 1
                
            elif P_test[ip][j][k] == 2:
            #elif X_test[ip][j][k] == 1:
                
                matrix_test_stress[vector_idx_2] = Y_test[ip,j,k]
                matrix_pred_stress[vector_idx_2] = predict[ip,j,k]
                vector_idx_2 = vector_idx_2 + 1
                
    max_real_fiber[ip]   = np.max(fiber_test_stress)
    max_pred_fiber[ip]   = np.max(fiber_pred_stress)
    
    error_max_fiber[ip]       = np.abs(max_real_fiber[ip]-max_pred_fiber[ip])
    error_rate_max_fiber[ip]  = error_max_fiber[ip] / max_real_fiber[ip]

    min_real_fiber[ip]   = np.min(fiber_test_stress)
    min_pred_fiber[ip]   = np.min(fiber_pred_stress)
    
    error_min_fiber[ip]  = np.abs(min_real_fiber[ip]-min_pred_fiber[ip])
    
    max_real_matrix[ip]  = np.max(matrix_test_stress)
    max_pred_matrix[ip]  = np.max(matrix_pred_stress)
    
    error_max_matrix[ip]      = np.abs(max_real_matrix[ip]-max_pred_matrix[ip])
    error_rate_max_matrix[ip] = error_max_matrix[ip] / max_real_matrix[ip]

    min_real_matrix[ip]  = np.min(matrix_test_stress)
    min_pred_matrix[ip]  = np.min(matrix_pred_stress)
    
    error_min_matrix[ip] = np.abs(min_real_matrix[ip]-min_pred_matrix[ip])

max_error_rate_fiber  = np.mean(error_rate_max_fiber)
max_error_rate_matrix = np.mean(error_rate_max_matrix)
max_error_fiber       = np.mean(error_max_fiber)
max_error_matrix      = np.mean(error_max_matrix)
min_error_fiber       = np.mean(error_min_fiber)
min_error_matrix      = np.mean(error_min_matrix)

print("max error average rate for fiber is:",  max_error_rate_fiber)
print("max error average rate for matrix is:", max_error_rate_matrix)
print("max error rate for fiber is:",  max_error_fiber)
print("max error rate for matrix is:", max_error_matrix)
print("min error rate for fiber is:",  min_error_fiber)
print("min error rate for matrix is:", min_error_matrix)

#fp = open('training result.txt','w')
result = np.savetxt('result_summary.txt',
                    (max_error_rate_fiber,max_error_rate_matrix,max_error_fiber,max_error_matrix,min_error_fiber,min_error_matrix,t1,t2,score[0]),
                    header='fiber max error average rate,matrix max error average rate,fiber max error rate,matrix max error rate,fiber min error rate,matrix min error rate,data_process,training_time,score')

# Plot CNN Model

tf.keras.utils.plot_model(model, to_file='model.png')

model.save('my_model.h5')