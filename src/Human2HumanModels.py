from cycleGAN import *

from tensorflow.keras.layers import Activation, Dense, Dropout, LeakyReLU as lrelu
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects

# Unique Activations
def humanActivation(x):
    xList = tf.unstack(x,axis=1)
    setNum = len(xList)//7
    yList = []
    for i in range(setNum):
        d = K.sqrt(K.square(xList[3+7*i])+K.square(xList[4+7*i])+K.square(xList[5+7*i])+K.square(xList[6+7*i]))
        yList.append(xList[0+7*i])
        yList.append(xList[1+7*i])
        yList.append(xList[2+7*i])
        yList.append(xList[3+7*i]/d)
        yList.append(xList[4+7*i]/d)
        yList.append(xList[5+7*i]/d)
        yList.append(xList[6+7*i]/d)
    for j in range(len(xList)%7):
        yList.append(xList[j + 7*setNum])
    return tf.stack(yList,axis=1)
def pepperActivation(x):
    xList = tf.unstack(x,axis=1)
    setNum = len(xList)//4
    yList = []
    for i in range(setNum):
        d = K.sqrt(K.square(xList[0+4*i])+K.square(xList[1+4*i])+K.square(xList[2+4*i])+K.square(xList[3+4*i]))
        yList.append(xList[0+4*i]/d)
        yList.append(xList[1+4*i]/d)
        yList.append(xList[2+4*i]/d)
        yList.append(xList[3+4*i]/d)
    return tf.stack(yList,axis=1)

get_custom_objects().update({'humanactivation':Activation(humanActivation)})
get_custom_objects().update({'pepperactivation':Activation(pepperActivation)})

def model1():
    # Need to identify this model
    descript = "Adult with layer encoder [18,14,10,14,18] and no dropout\nKid same as Adult\nUsing 10 latent features"

    # Build the two autoencoders
    encodeH = [
        Dense(14,activation=lrelu(alpha=0.01)),
        Dense(10,activation=lrelu(alpha=0.01)),
        ]
    decodeH = [
        Dense(14,activation=lrelu(alpha=0.01)),
        Dense(18),
        ]

    encodeP = [
        Dense(14,activation=lrelu(alpha=0.01)),
        Dense(10,activation=lrelu(alpha=0.01)),
        ]
    decodeP = [
        Dense(14,activation=lrelu(alpha=0.01)),
        Dense(18),
        ]

    # Create student-teacher model
    model = CycleGANModel(18,18)
    model.createModel1(encodeH,decodeH)
    model.createModel2(encodeP,decodeP)

    model.compile()
    return model, descript

def model2():
    # Need to identify this model
    descript = "Adult with layer encoder [18,16,14,12,10,12,14,16,18] and no dropout\nKid same as Adult\nUsing 10 latent features"

    # Build the two autoencoders
    encodeH = [
        Dense(16,activation=lrelu(alpha=0.01)),
        Dense(14,activation=lrelu(alpha=0.01)),
        Dense(12,activation=lrelu(alpha=0.01)),
        Dense(10,activation=lrelu(alpha=0.01)),
        ]
    decodeH = [
        Dense(12,activation=lrelu(alpha=0.01)),
        Dense(14,activation=lrelu(alpha=0.01)),
        Dense(16,activation=lrelu(alpha=0.01)),
        Dense(18),
        ]

    encodeP = [
        Dense(16,activation=lrelu(alpha=0.01)),
        Dense(14,activation=lrelu(alpha=0.01)),
        Dense(12,activation=lrelu(alpha=0.01)),
        Dense(10,activation=lrelu(alpha=0.01)),
        ]
    decodeP = [
        Dense(12,activation=lrelu(alpha=0.01)),
        Dense(14,activation=lrelu(alpha=0.01)),
        Dense(16,activation=lrelu(alpha=0.01)),
        Dense(18),
        ]

    # Create student-teacher model
    model = CycleGANModel(18,18)
    model.createModel1(encodeH,decodeH)
    model.createModel2(encodeP,decodeP)

    model.compile()
    return model, descript

def model3():
    # Need to identify this model
    descript = "Adult with layer encoder [18,14,10,6,10,14,18] and no dropout\nKid same as Adult\nUsing 6 latent features"

    # Build the two autoencoders
    encodeH = [
        Dense(14,activation=lrelu(alpha=0.01)),
        Dense(10,activation=lrelu(alpha=0.01)),
        Dense(6,activation=lrelu(alpha=0.01)),
        ]
    decodeH = [
        Dense(10,activation=lrelu(alpha=0.01)),
        Dense(14,activation=lrelu(alpha=0.01)),
        Dense(18),
        ]

    encodeP = [
        Dense(14,activation=lrelu(alpha=0.01)),
        Dense(10,activation=lrelu(alpha=0.01)),
        Dense(6,activation=lrelu(alpha=0.01)),
        ]
    decodeP = [
        Dense(10,activation=lrelu(alpha=0.01)),
        Dense(14,activation=lrelu(alpha=0.01)),
        Dense(18),
        ]

    # Create student-teacher model
    model = CycleGANModel(18,18)
    model.createModel1(encodeH,decodeH)
    model.createModel2(encodeP,decodeP)

    model.compile()
    return model, descript

mList = [model1,model2,model3]
mNames = ["Small","Large","Fewer"]
