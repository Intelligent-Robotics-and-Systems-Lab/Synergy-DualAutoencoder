import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Lambda

from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

class CycleGANModel:
    ''' Class for dealing with the student model
    '''
    # Create three input instances
    input1 = None
    input2 = None

    # Track the four input layers
    encode1Layers = None
    encode2Layers = None
    decode1Layers = None
    decode2Layers = None

    # Number of layers per part
    encode1Num = None
    encode2Num = None
    decode1Num = None
    decode2Num = None
    embedNum = None

    # The three required los functions
    loss1 = None
    loss2 = None
    loss12 = None
    loss21 = None
    loss121 = None
    loss212 = None

    # Stores the compiled models
    Mone = None
    Mtwo = None
    Mone2two = None
    Mtwo2one = None
    Mone2two2one = None
    Mtwo2one2two = None
    lossModel = None

    # Stores knowledge
    history = None
    validating = False
    checkDir = "./training"
    checkPath = "./training/cp-{epoch:05d}.ckpt"

    def __init__(self,inSize1,inSize2):
        self.input1 = Input(shape=inSize1,name="Input1")
        self.input2 = Input(shape=inSize2,name="Input2")

    def createModel1(self,encodeLayers,decodeLayers):
        '''Adds a new layer to the existing model 1.
        encodeLayers and decodeLayers are an array of keras.layers used for the encoding and decode of model 1.
        '''
        # Check embedded size
        if (self.embedNum is None):
            self.embedNum = encodeLayers[-1].units
        elif (self.embedNum != encodeLayers[-1].units):
            print("The size of the embeded layer differs from the previously defined embedded layer.")
            print("The last layer of the encodeLayers for both models should be the same.")
            return 1

        self.encode1Layers = encodeLayers
        self.encode1Num = len(encodeLayers)
        self.decode1Layers = decodeLayers
        self.decode1Num = len(decodeLayers)
        return 0

    def createModel2(self,encodeLayers,decodeLayers):
        '''Adds a new layer to the existing model 2.
        encodeLayers and decodeLayers are an array of keras.layers used for the encoding and decode of model 2.
        '''
        # Check embedded size
        if (self.embedNum is None):
            self.embedNum = encodeLayers[-1].units
        elif (self.embedNum != encodeLayers[-1].units):
            print("The size of the embeded layer differs from the previously defined embedded layer.")
            print("The last layer of the encodeLayers for both models should be the same.")
            return 1

        self.encode2Layers = encodeLayers
        self.encode2Num = len(encodeLayers)
        self.decode2Layers = decodeLayers
        self.decode2Num = len(decodeLayers)
        return 0

    def viewModel(self,encoder=-1,decoder=0):
        ''' Method for viewing the models once compiled.
        encoder is either 1 or 2 for using model 1 or 2 as input.
        decoder is either 1 or 2 for using model 1 or 2 as output.
        if decoder is 0, using the autoencoder model
        if encoder or decoder both -1, then viewing the training model.
        '''
        try:
            if (encoder == -1) or (decoder == -1):
                self.lossModel.summary()
            elif (encoder == 1) and (decoder == 0):
                self.Mone.summary()
            elif (encoder == 1) and (decoder == 1):
                self.Mone2two2one.summary()
            elif (encoder == 1) and (decoder == 2):
                self.Mone2two.summary()
            elif (encoder == 2) and (decoder == 0):
                self.Mtwo.summary()
            elif (encoder == 2) and (decoder == 1):
                self.Mtwo2one.summary()
            elif (encoder == 2) and (decoder == 2):
                self.Mtwo2one2two.summary()
            else:
                print("Not a valid pair")
        except:
            print("Unable to view models until models are compiled")

    def compile(self,weights=[1,1,1,1,1,1]):
        # Models A to A
        m1 = self.input1
        for count in range(self.encode1Num):
            m1 = self.encode1Layers[count](m1)
        for count in range(self.decode1Num):
            m1 = self.decode1Layers[count](m1)
        self.Mone = Model(self.input1,m1)

        m2 = self.input2
        for count in range(self.encode2Num):
            m2 = self.encode2Layers[count](m2)
        for count in range(self.decode2Num):
            m2 = self.decode2Layers[count](m2)
        self.Mtwo = Model(self.input2,m2)

        # Models A to B
        m12 = self.input1
        for count in range(self.encode1Num):
            m12 = self.encode1Layers[count](m12)
        for count in range(self.decode2Num):
            m12 = self.decode2Layers[count](m12)
        self.Mone2two = Model(self.input1,m12)

        m21 = self.input2
        for count in range(self.encode2Num):
            m21 = self.encode2Layers[count](m21)
        for count in range(self.decode1Num):
            m21 = self.decode1Layers[count](m21)
        self.Mtwo2one = Model(self.input2,m21)

        # Models A to B to A
        m121 = self.input1
        for count in range(self.encode1Num):
            m121 = self.encode1Layers[count](m121)
        for count in range(self.decode2Num):
            m121 = self.decode2Layers[count](m121)
        for count in range(self.encode2Num):
            m121 = self.encode2Layers[count](m121)
        for count in range(self.decode1Num):
            m121 = self.decode1Layers[count](m121)
        self.Mone2two2one = Model(self.input1,m121)

        m212 = self.input2
        for count in range(self.encode2Num):
            m212 = self.encode2Layers[count](m212)
        for count in range(self.decode1Num):
            m212 = self.decode1Layers[count](m212)
        for count in range(self.encode1Num):
            m212 = self.encode1Layers[count](m212)
        for count in range(self.decode2Num):
            m212 = self.decode2Layers[count](m212)
        self.Mtwo2one2two = Model(self.input2,m212)

        # Training model
        self.loss1 = Lambda(lambda x:self.lossFunction(*x),name="Model1Loss")([self.input1,m1])
        self.loss2 = Lambda(lambda x:self.lossFunction(*x),name="Model2Loss")([self.input2,m2])
        self.loss12 = Lambda(lambda x:self.lossFunction(*x),name="Model12Loss")([self.input2,m12])
        self.loss21 = Lambda(lambda x:self.lossFunction(*x),name="Model21Loss")([self.input1,m21])
        self.loss121 = Lambda(lambda x:self.lossFunction(*x),name="Model121Loss")([self.input1,m121])
        self.loss212 = Lambda(lambda x:self.lossFunction(*x),name="Model212Loss")([self.input2,m212])
        self.lossModel = Model([self.input1,self.input2],[m1,m2,m12,m21,m121,m212])
        self.lossModel.add_loss(weights[0]*self.loss1)
        self.lossModel.add_loss(weights[1]*self.loss2)
        self.lossModel.add_loss(weights[2]*self.loss12)
        self.lossModel.add_loss(weights[3]*self.loss21)
        self.lossModel.add_loss(weights[4]*self.loss121)
        self.lossModel.add_loss(weights[5]*self.loss212)
        self.lossModel.compile(optimizer='adam',loss=[None,None,None,None,None,None])

    def lossFunction(self,y_true,y_pred,args=None):
        if args is None:
            delta = 0.6
        else:
            delta = args[0]
        loss = Huber(delta=delta)
        return loss(y_true,y_pred)

    def loadWeights(self,ckpt=None):
        '''If already began training model, can reload from a checkpoint.
        Use after model is compiled
        '''
        if ckpt is None:
            ckpt = tf.train.latest_checkpoint(self.checkDir)
        self.lossModel.load_weights(ckpt)

    def fit(self,x1,x2,val1=None,val2=None,epochs=1000,batch_size=10,verbose=0):
        ''' Trains two autoencoders using the Student-Teacher Model.
        '''

        callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = self.checkPath,
            verbose=0,
            save_weights_only=True,
            period=100)

        if (val1 is None) and (val2 is None):
            self.validating = False
            self.history = self.lossModel.fit([x1,x2],None,callbacks=[callback],epochs=epochs,batch_size=batch_size,verbose=verbose)
        else:
            self.validating = True
            self.history = self.lossModel.fit([x1,x2],None,validation_data=([val1,val2],None),callbacks=[callback],epochs=epochs,batch_size=batch_size,verbose=verbose)

        return self.history

    def evaluate(self,x1,x2):
        pred11 = self.predict(x1,1,0)
        pred121 = self.predict(x1,1,1)
        pred12 = self.predict(x1,1,2)
        pred22 = self.predict(x2,2,0)
        pred21 = self.predict(x2,2,1)
        pred212 = self.predict(x2,2,2)
        error11 = mse(x1,pred11)
        error121 = mse(x1,pred121)
        error12 = mse(x2,pred12)
        error22 = mse(x2,pred22)
        error21 = mse(x1,pred21)
        error212 = mse(x2,pred212)
        return error11,error121,error12,error22,error21,error212

    def predict(self,x,encoder=None,decoder=0):
        if encoder is None:
            if x.shape[1:] == self.input1.shape[1:]:
                encoder = 1
            elif x.shape[1:] == self.input2.shape[1:]:
                encoder = 2
            else:
                print("Could not use either of the models.\nRemember that input should either be shape (n,{}) or (n,{})".format(self.input1.shape[1:],self.input2.shape[1:]))

        if (encoder == 1) and (decoder == 0):
            return self.Mone.predict(x)
        elif (encoder == 1) and (decoder == 1):
            return self.Mone2two2one.predict(x)
        elif (encoder == 1) and (decoder == 2):
            return self.Mone2two.predict(x)
        elif (encoder == 2) and (decoder == 0):
            return self.Mtwo.predict(x)
        elif (encoder == 2) and (decoder == 1):
            return self.Mtwo2one.predict(x)
        elif (encoder == 2) and (decoder == 2):
            return self.Mtwo2one2two.predict(x)
        else:
            print("Did not pick a valid encoder-decoder pair")
            return "Invalid"

    def plotHistory(self):
        '''Plots the history of the three loss functions
        '''
        if self.validating:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.plot(self.history.history['loss'])
            ax1.set_title("Training Loss")
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax2 = fig.add_subplot(122)
            ax2.plot(self.history.history['val_loss'])
            ax2.set_title("Validation Loss")
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(self.history.history['loss'])
            ax.set_title("Training Loss")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
        return fig
