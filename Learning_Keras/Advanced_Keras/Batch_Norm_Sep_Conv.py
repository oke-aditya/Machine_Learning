
# - Batch normalization is a type of layer (BatchNormalization in Keras) introduced
# in 2015 by Ioffe and Szegedy. 
# - It can adaptively normalize data even as the mean and variance change over time during training. It works by internally maintaining an exponential moving average of the batch-wise mean and variance of the data seen during
# training. 
# - The main effect of batch normalization is that it helps with gradient propagation—
# much like residual connections—and thus allows for deeper networks. 

# - The BatchNormalization layer is typically used after a convolutional or densely
# connected layer:
# 
# - The BatchNormalization layer takes an axis argument, which specifies the feature
# axis that should be normalized. 
# - This argument defaults to -1, the last axis in the input
# tensor. 
# - This is the correct value when using Dense layers, Conv1D layers, RNN layers,
# and Conv2D layers with data_format set to "channels_last". 
# - But in the niche use case of Conv2D layers with data_format set to "channels_first", the features axis is axis 1;
# the axis argument in BatchNormalization should accordingly be set to 1.


conv_model.add(layers.Conv2D(32, 3, activation = 'relu'))
conv_model.add(BatchNormalization())

dense_model.add(layers.Dense(32, activation = 'relu'))
dense_model.add(BatchNormalization())


# # Depthwise Separable Convolution

# - The depthwise separable convolution layer does
# (SeparableConv2D). 
# - This layer performs a spatial convolution on each channel of its
# input, independently, before mixing output channels via a pointwise convolution (a
# 1 × 1 convolution),

# - This is equivalent to separating the learning of spatial features and the learning of channel-wise features, which makes a lot of
# sense if you assume that spatial locations in the input are highly correlated, but different
# channels are fairly independent. 
# - It requires significantly fewer parameters and
# involves fewer computations, thus resulting in smaller, speedier models. And because
# it’s a more representationally efficient way to perform convolution, it tends to learn
# better representations using less data, resulting in better-performing models

# - These advantages become especially important when you’re training small models
# from scratch on limited data.
# - When it comes to larger-scale models, depthwise separable convolutions are the basis
# of the Xception architecture, a high-performing convnet that comes packaged with
# Keras.


from keras.models import Sequential, Model
from keras import layers



height = 64
width = 64
channels = 3
num_classes = 10



model = Sequential()
model.add(layers.SeparableConv2D(32, 3, activation = 'relu', input_shape = (height, width, channels)))
model.add(layers.SeperableConv2D(64, 3, activation = 'relu'))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(64, 3, activation='relu'))
model.add(layers.SeparableConv2D(128, 3, activation='relu'))
model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['acc'])



model.summary()

