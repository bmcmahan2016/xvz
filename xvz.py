import tensorflow as tf
import numpy as np 
import os
import tensorflow.keras.layers as tfl 
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Resizing, Normalization
from glob import glob
import random
import pdb
import tensorflow_addons as tfa  # needed for weight decay
import io
import datetime
import tqdm

AUTOTUNE = tf.data.experimental.AUTOTUNE  # performance optimization

import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
)
def normalize_img(image, labels):
    tmp = Resizing(227, 227)
    return tmp(tf.cast(image, tf.float32) / 255.0), labels

'''for image, label in ds_train.take(1):
    print("image:", image.numpy())
    print("label:", label.numpy())
'''
ds_train = ds_train.map(normalize_img)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(512)
ds_train = ds_train.prefetch(AUTOTUNE)


####################################################
# VISUALIZATION CALLBACKS
####################################################

logdir = "logs/train_data/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir)

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  buf = io.BytesIO()  # Save the plot to a PNG in memory.
  plt.savefig(buf, format='png')
  plt.close(figure)
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)   # Convert PNG buffer to TF image
  image = tf.expand_dims(image, 0)  # Add the batch dimension
  return image

def get_layers(model, name='conv'):
    '''takes as input a keras model and returns a list containing the 
    indices of all convolutional layers. Other layer indices can be 
    obtained by changing the name argument'''
    idxs = []
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if name in layer.name:  # this is a convolutional layer
            idxs.append(i)    # i indexes a convolutional layer
    return idxs

def image_grid(img, special_tittle=False):
  """
  Creates a grid of images, with unlimited number of rows and up to 10 image per row.

  Arguments
    imgs -- tf tensor containing the image data to be plotted in the grid, has 
    shape (num_images, image_height, image_width, input_channels)
    special_tittle -- can be set to a string that will be used as the tittle 
    for the first subplot
  
  Returns
    figure -- matplotlib figure object
  """
  
  figure = plt.figure(figsize=(16,24))  # Create a figure to contain the plot
  out_channels, im_height, im_width, in_channels = img.shape  # extract dimesnions from image
  for i in range(out_channels):
    # Start next subplot.
    if special_tittle:  # use a special tittle
        if i == 0:
            sub_tittle = special_tittle
        else:
            sub_tittle = "#" + str(i-1)
    else:  # don't use a special tittle
        sub_tittle = "#" + str(i)
    plt.subplot(np.minimum(10, out_channels), int(out_channels/10) + 1, i + 1, title=sub_tittle)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # we want to plot an image of the weights for each output channel (neuron)
    #print("shape of wieghts:", img.shape)       
    plt.imshow(img[i,:,:,0], cmap=plt.cm.binary)  # currently only using first image channel

  return figure

# done!
class VisualizeConvWeights(tf.keras.callbacks.Callback):
    """
    Saves the convolutional weights to the tensorboard at the end of 
    each training epoch
    """
    def on_epoch_end(self, epoch, logs=None):
        with file_writer.as_default():
            #print("convolutional layers:", get_conv_layers(self.model))
            for curr_layer in get_layers(self.model, name='conv'):
                # extract weights of current convolutional layer
                this_layer = self.model.get_layer(index=curr_layer)
                weights, biases = this_layer.get_weights()
                img = tf.transpose(weights, perm=[3,0,1,2])  # img is (out_channels, image_heigh, image_width, in_channels)
                figure = image_grid(img)
                img_title = "Epoch #" + str(epoch) + "Layer (Conv2D) #" + str(curr_layer)
                tf.summary.image(img_title, plot_to_image(figure), step=0)

# done!
class VisualizeDenseWeights(tf.keras.callbacks.Callback):
    """
    Saves the fully connected layer weights to the tensorboard at the end of 
    each training epoch
    """
    def on_epoch_end(self, epoch, logs=None):
        with file_writer.as_default():
            #print("convolutional layers:", get_conv_layers(self.model))
            for curr_layer in get_layers(self.model, name='dense'):
                # extract weights of current dense layer
                this_layer = self.model.get_layer(index=curr_layer)
                weights, biases = this_layer.get_weights()
                #pdb.set_trace()
                img = tf.expand_dims(weights, axis=0)
                img = tf.expand_dims(img, axis=-1)
                figure = image_grid(img)
                img_title = "Epoch #" + str(epoch) + "Layer (Dense) #" + str(curr_layer)
                tf.summary.image(img_title, plot_to_image(figure), step=0)

# still editing
class VisualizeActivations(tf.keras.callbacks.Callback):
    '''plots the activations of intermediate layers'''
    def on_epoch_end(self, epoch, logs=None):
        with file_writer.as_default():
            print("Saving activations to tensorboard!")
            for curr_layer in tqdm.tqdm(get_layers(self.model, name='conv')):
                # extract weights of current dense layer
                this_layer = self.model.get_layer(index=curr_layer)
                activations = tf.keras.Model(inputs=self.model.inputs, outputs=this_layer.output)
                for image, label in ds_train.take(1):
                    plt.figure()
                    plt.imshow(image[0,:,:,0].numpy())
                    plt_tittle = "Digit: " + str(label[0].numpy())
                    plt.title(plt_tittle)
                features = activations.predict(image)
                features = tf.transpose(features, perm=[3,1,2,0])       #feature_maps has shape (n_neurons, height, width, batch_size)
                features = tf.expand_dims(features, axis=3)[:,:,:,:,0]  # add channel dimension and look at only single input from batch
                _, height, width, _ = features.shape
                original_img = tf.image.resize(image[0], (height, width))
                original_img = tf.expand_dims(original_img, axis=0)   # needed to match dimensions
                features = tf.concat([original_img, features], 0)  # add input image next to layer activations
                figure = image_grid(features, )
                # prepend input image that gave rise to this data
                #pdb.set_trace()
                figure = image_grid(features, special_tittle="Input Image")
                img_title = "Epoch #" + str(epoch) + "Layer #" +  str(curr_layer) + "activations"
                tf.summary.image(img_title, plot_to_image(figure), step=0)


####################################################

# todo: add local response normalization
# AlexNet Architecture
ones = tf.keras.initializers.Constant(value=1)
initial_weights = tf.keras.initializers.RandomNormal(
        mean=0.0,
        stddev=0.01
)

# simple implementation of AlexNet
def AlexNet():
	"""
	Generates the AlexNet model
        still need to implement the local response normalization
	"""
	# inputs are 227x227x3

	model = tf.keras.Sequential([
                    #Normalization(input_shape=(227, 227, 3)),
      	            tf.keras.layers.Conv2D(96, (11, 11), strides=(4,4), padding="valid", activation='relu', kernel_initializer=initial_weights, input_shape=(227,227,1)),  # first conv layer
                    tfl.BatchNormalization(),
	            tfl.MaxPool2D(pool_size=(3,3), strides=(2,2)),
	            tf.keras.layers.Conv2D(256, (5, 5), padding="same", activation='relu', bias_initializer=ones, kernel_initializer=initial_weights),                   # second conv layer
                    tfl.BatchNormalization(),
	            tfl.MaxPool2D(pool_size=(3,3), strides=(2,2)),
	            tf.keras.layers.Conv2D(384, (3, 3), padding="same", activation='relu', kernel_initializer=initial_weights),                   # third conv layer
                    tfl.BatchNormalization(),
	            tf.keras.layers.Conv2D(384, (3, 3), padding="same", activation='relu', bias_initializer=ones, kernel_initializer=initial_weights),                   # fourth conv layer
                    tfl.BatchNormalization(),
	            tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation='relu', bias_initializer=ones, kernel_initializer=initial_weights),                   # fifth conv layer
                    tfl.BatchNormalization(),
	            tfl.MaxPool2D(pool_size=(3,3), strides=(2,2)),
	            # Flatten network
	            tfl.Flatten(),
	            tfl.Dense(4096, activation='relu', bias_initializer=ones, kernel_initializer=initial_weights),                                        # first dense layer
                    tfl.Dropout(0.5),
	            tfl.Dense(4096, activation='relu', bias_initializer=ones, kernel_initializer=initial_weights),                                        # second dense layer
                    tfl.Dropout(0.5),
	            tfl.Dense(10, activation='softmax', kernel_initializer=initial_weights)                                      # final dense layer followed by softmax
	        ])
	return model
model = AlexNet()
#model = tf.keras.models.load_model('AlexNet')
print("model passed all unit tests !")
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.01,
        decay_steps = 30000,
        decay_rate=0.1,
        staircase=True)

#opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
opt = tfa.optimizers.SGDW(
        weight_decay=0.005,
        momentum=0.9,
        learning_rate=lr_schedule
)
model.compile(optimizer=opt,
	          loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
model.summary()

# create a tensorboard callback
logs = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logs, 
        write_images=True,
        embeddings_freq=3,
        profile_batch='20, 40',
)

model.fit(x=ds_train, 
	      epochs=2,
              callbacks=[VisualizeActivations()])
model.save("AlexNet")
