#%% Description
#
# PC_SR_C refers to plate with circular cutout model with controlled spatial randomness, using DiNN-NC framework
#
#%% Import Modules
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time

#%% Section To Run With GPU

# Choose GPU to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU ID to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0";

Config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
Config.gpu_options.allow_growth=True

#%% Load Data and Preprocessing

# Define Cartesian Map grid size (user defined) -- a size matching targetting geometry size is recommended
rectangle_x = 0.01
rectangle_y = 0.01

# Total number of dataset (user defined)
data_set = 1000

# Uniform dimension for input data (Line 57-60 import data of this size)
n_x = 79
n_y = 79

grid_dist_x = rectangle_x / n_x
grid_dist_y = rectangle_y / n_y

m_x = int(n_x + 1)
m_y = int(n_y + 1)

# Initialize data matrices
data_cart_new   = np.zeros((data_set, m_x, m_y))     # Geometry matrix
one_cart_new    = np.zeros((data_set, m_x, m_y))     # Uni-geometry-matrix for clean module
data_stress_new = np.zeros((data_set, m_x, m_y))     # Stress matrix
one_stress_new  = np.zeros((data_set, m_x, m_y))     # Uni-stress-matrix for clean module (same as geometry Uni-geometry matrix)
data_property   = np.zeros((data_set, m_x, m_y))     # Property matrix

# Record time
start_1 = time()

# Read data from ABAQUS Analysis result files
for k in range(data_set):
    
    idx = str(k+1)
    
    # Load Cartesian Map data (dimension should follow n_x by n_y on line 31-32)
    # These file paths are subject to change when using in local directory
    txt_cart    = 'C:\\Temp_Abaqus\\micro_meter_model\\random_hole_hollow_four\\Composite_uniform_SDF_Cart_' + idx + '.dat'
    txt_stress  = 'C:\\Temp_Abaqus\\micro_meter_model\\random_hole_hollow_four\\Composite_uniform_Stress_Cart_' + idx + '.dat'
    one_cart    = 'C:\\Temp_Abaqus\\micro_meter_model\\random_hole_hollow_four\\One_SDF_Cart_' + idx + '.dat'
    one_stress  = 'C:\\Temp_Abaqus\\micro_meter_model\\random_hole_hollow_four\\One_Stress_Cart_' + idx + '.dat'

    data_cart   = np.loadtxt(txt_cart)
    data_stress = np.loadtxt(txt_stress)
    one_cart    = np.loadtxt(one_cart)
    one_stress  = np.loadtxt(one_stress)

    [m,n] = data_cart.shape
    
    # Reshape input data into matrices
    for i in range(m): 
        x = int(round(data_cart[i][0] / grid_dist_x))
        y = int(round(data_cart[i][1] / grid_dist_y))
        data_cart_new[k][y][x] = data_cart[i][2]
        one_cart_new[k][y][x]  = one_cart[i][2]
        one_stress_new[k][y][x]  = one_stress[i][2]
        data_stress_new[k][y][x] = data_stress[i][2]
        #data_property[k][y][x]   = data_stress[i][3]

#%% Data Random Split     
X_train, X_test, Y_train, Y_test, One_cart_train, One_cart_test, One_stress_train, One_stress_test = train_test_split(data_cart_new, data_stress_new, one_cart_new,  one_stress_new,  test_size=0.2, random_state=47)
X_test,  X_cv,   Y_test,  Y_cv,   One_cart_test,  One_cart_cv,   One_stress_test,  One_stress_cv   = train_test_split(X_test,        Y_test,          One_cart_test, One_stress_test, test_size=0.5, random_state=47)

# Reshape data into matrices for neural network training (#sample, X, Y, Feature)
One_cart_train   = tf.reshape(One_cart_train,   [-1, m_x, m_y, 1])
One_stress_train = tf.reshape(One_stress_train, [-1, m_x, m_y, 1])
One_cart_test    = tf.reshape(One_cart_test,    [-1, m_x, m_y, 1])
One_stress_test  = tf.reshape(One_stress_test,  [-1, m_x, m_y, 1])
One_cart_cv      = tf.reshape(One_cart_cv,      [-1, m_x, m_y, 1])
One_stress_cv    = tf.reshape(One_stress_cv,    [-1, m_x, m_y, 1])

input_train      = tf.reshape(X_train, [-1, m_x, m_y, 1])
output_train     = tf.reshape(Y_train, [-1, m_x, m_y, 1])
input_cv  = tf.reshape(X_cv, [-1, m_x, m_y, 1])
output_cv = tf.reshape(Y_cv, [-1, m_x, m_y, 1])
input_test  = tf.reshape(X_test, [-1, m_x, m_y, 1])
output_test = tf.reshape(Y_test, [-1, m_x, m_y, 1])

# Take the reference geometry and stress contour (mean geometry in this code)
sdf_ave  = tf.reduce_mean(input_train, 0)
sdf_ave  = tf.reshape(sdf_ave, [-1, m_x, m_y, 1])
stress_ave = tf.reduce_mean(output_train, 0)
stress_ave = tf.reshape(stress_ave, [-1, m_x, m_y, 1])

# Calculate the geometry and stress different contour for Training set, Cross-validation set and Test set
input_train_new  = input_train - sdf_ave
output_train_new = output_train - stress_ave
[s1,s2,s3,s4]    = input_train.shape
# Repeat the mean contour to match with size of training data
stress_ave_train = tf.keras.backend.repeat_elements(stress_ave, rep=s1, axis=0)  

input_cv_new  = input_cv - sdf_ave
[c1,c2,c3,c4] = input_cv.shape
# Repeat the mean contour to match with size of cross-validation data
stress_ave_cv = tf.keras.backend.repeat_elements(stress_ave, rep=c1, axis=0)

input_test_new    = input_test - sdf_ave
[te1,te2,te3,te4] = input_test.shape
# Repeat the mean contour to match with size of testing data
stress_ave_test   = tf.keras.backend.repeat_elements(stress_ave, rep=te1, axis=0) 

#%% Normalization Module

max_sdf     = np.max(input_train_new)
max_stress  = np.max(output_train_new)
min_sdf     = np.min(input_train_new)
min_stress  = np.min(output_train_new)

# Min-max normalization
input_train_new = (input_train_new - min_sdf) / (max_sdf - min_sdf) # min-max norm
input_train_new = tf.math.multiply(input_train_new, One_cart_train)
input_cv_new    = (input_cv_new - min_sdf)    / (max_sdf - min_sdf)
input_cv_new    = tf.math.multiply(input_cv_new, One_cart_cv)
input_test_new  = (input_test_new - min_sdf)  / (max_sdf - min_sdf)
input_test_new  = tf.math.multiply(input_test_new, One_cart_test)

# Take the preprocessing time and start record training time
t1 = time() - start_1
start_2 = time()

#%% Defining Neural Network

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

def dense_block(x,filt,names):
    
    y = tf.keras.layers.Dense(filt, activation='linear',
                              kernel_initializer='he_normal', use_bias=False,
                              name=names)(x)
    
    return y

#%% Encoder-Decoder Neural Network Structure

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

#%% Training The DiNN framework

# Set training optimizer
sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.6, nesterov=True)

# Compile the model
model.compile(optimizer=sgd, loss=tf.keras.losses.mean_squared_error, metrics=['accuracy'])

epoch = 80
# Fit (Train) the model
history = model.fit([input_train_new, stress_ave_train, One_stress_train], output_train, batch_size=256, epochs=epoch, 
                    steps_per_epoch=40, validation_data=([input_cv_new, stress_ave_cv, One_stress_cv], output_cv))

# Evaluate the model on test set
predict = model.predict([input_test_new, stress_ave_test, One_stress_test])

score = model.evaluate([input_test_new, stress_ave_test, One_stress_test], output_test, verbose=1)
print('\n', 'Test accuracy', score)

# Record Neural Network Training and Prediction Time
t2 = time() - start_2 

#%% Generating history plots of training

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

#%% Plot and generate graphs for test samples 

# Geometry average plot
sdf_ave_plot = sdf_ave[0,:,:,0]

fig0_sdf_ave = plt.figure()
plt.title('SDF average')
plt.imshow(sdf_ave_plot, cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()
fig0_sdf_ave.savefig('sdf_ave.png')

# stress average plot
stress_ave_plot = stress_ave[0,:,:,0]

fig0_stress_ave = plt.figure()
plt.title('stress average')
plt.imshow(stress_ave_plot, cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()
fig0_stress_ave.savefig('stress_ave.png')

# The first test set
X_test_1 = input_test_new[0, :, :, 0]
Y_test_1 = output_test[0, :, :, 0]

fig1_sdf = plt.figure()
plt.title('SDF contour')
plt.imshow(X_test_1, cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()
fig1_sdf.savefig('SDF_1.png')

fig1_test = plt.figure()
plt.title('Stress contour')
plt.imshow(Y_test_1, cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()
fig1_test.savefig('Test_1.png')

predict_1 = predict[0, :, :, 0]
fig1_pred=plt.figure()
plt.title('Trained stress contour')
plt.imshow(predict_1, cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()
fig1_pred.savefig('Predict_1.png')

# The second test set
X_test_2 = input_test_new[3, :, :, 0]
Y_test_2 = output_test[3, :, :, 0]

fig2_sdf = plt.figure()
plt.title('SDF contour')
plt.imshow(X_test_2, cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()
fig2_sdf.savefig('SDF_2.png')

fig2_test = plt.figure()
plt.title('Stress contour')
plt.imshow(Y_test_2, cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()
fig2_test.savefig('Test_2.png')

predict_2 = predict[3,:,:,0]
fig2_pred=plt.figure()
plt.title('Trained stress contour')
plt.imshow(predict_2, cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()
fig2_pred.savefig('Predict_2.png')

# The third test set
X_test_3 = input_test_new[8, :, :, 0]
Y_test_3 = output_test[8, :, :, 0]

fig3_sdf = plt.figure()
plt.title('SDF contour')
plt.imshow(X_test_3, cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()
fig3_sdf.savefig('SDF_3.png')

fig3_test=plt.figure()
plt.title('Stress contour')
plt.imshow(Y_test_3, cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()
fig3_test.savefig('Test_3.png')

predict_3 = predict[8, :, :, 0]
fig3_pred=plt.figure()
plt.title('Trained stress contour')
plt.imshow(predict_3, cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()
fig3_pred.savefig('Predict_3.png')

# Output the stress difference
Y_diff = Y_test_1 - predict_1
fig_diff = plt.figure()
plt.title('Stress_difference')
plt.imshow(Y_diff,cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()
fig_diff.savefig('Stress_difference.png')

#%% Plot outputs of individual layers

# (1) plot the original one
input_image = input_train[0, :, :, 0]

plt.figure()
plt.title('input contour')
plt.imshow(input_image, cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()

# (2) plot individual layer outputs (user can define the layer they want to output)
layer_names = ['conv1','conv2','conv3','deconv0','deconv1','deconv2','deconv3']

for layer_name in layer_names:
   
    inter_layer_model = tf.keras.models.Model(inputs=model.input,
                                          outputs=model.get_layer(layer_name).output)    
    inter_output = inter_layer_model.predict([input_test_new, stress_ave_train, One_stress_test])    
    ran = inter_output.shape[3]
    
    for j in range(ran):
        plot_inter_output = inter_output[0,:,:,j]
        layer_name_new = layer_name + str(j)
        
        fig = plt.figure()
        plt.title(layer_name_new)
        plt.imshow(plot_inter_output, cmap='rainbow')
        plt.colorbar()
        plt.grid(True) 
        
        name_layer = layer_name_new + '.png'
        fig.savefig(name_layer)
    
# (3) plot the real stress contour
output_image = output_train[0,:,:,0]

fig2 = plt.figure()
plt.title('stress contour')
plt.imshow(output_image, cmap='rainbow')
plt.colorbar()
plt.grid(True)
plt.show()

fig2.savefig('real_stress.png')

#%% Evaluate prediction errors of DiNN

predict_output = predict[:, :, :, 0]
[p1,p2,p3] = predict_output.shape

# Initialize error matrices
max_real  = np.zeros(p1) 
min_real  = np.zeros(p1)

max_pred  = np.zeros(p1)
min_pred  = np.zeros(p1)

error_max = np.zeros(p1)
error_min = np.zeros(p1)
error_rate_max = np.zeros(p1)

# Loop through test sample to calculate average prediction error
for ip in range(p1):
                
    max_real[ip]   = np.max(Y_test[ip,:,:])
    max_pred[ip]   = np.max(predict[ip,:,:])
    
    error_max[ip]       = np.abs(max_real[ip] - max_pred[ip])
    error_rate_max[ip]  = error_max[ip] / max_real[ip]

    min_real[ip]   = np.min(Y_test[ip,:,:])
    min_pred[ip]   = np.min(predict[ip,:,:])
    error_min[ip]  = np.abs(min_real[ip] - min_pred[ip])

max_error_rate   = np.mean(error_rate_max)
max_error  = np.mean(error_max)
min_error  = np.mean(error_min)

# Print out prediction error results
print("max error average rate is:",  max_error_rate)
print("max error rate is:",  max_error)
print("min error rate is:",  min_error)

#%% Save the training results 
result = np.savetxt('result_summary.txt',
                    (max_error_rate,max_error,min_error,t1,t2,score[0]),
                    header='max error average rate, max error rate, min error rate, data_process, training, score')

# Plot CNN Model
tf.keras.utils.plot_model(model, to_file='model.png')
model.save('my_model.h5')