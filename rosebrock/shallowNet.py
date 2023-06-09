from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras import backend as K

class ShallowNet:
  @staticmethod
  def build(width, height, depth, classes):
    inputShape = (height, width, depth)
    if K.image_data_format() == "channels_first":
      inputShape = (depth, height, width)
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation("softmax"))
    return model
    