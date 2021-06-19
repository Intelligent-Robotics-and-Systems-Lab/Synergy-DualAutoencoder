import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Lambda

from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

class SindyDualAutoencoderModel:
    ''' Class for dealing with the sindy dual autoencoder model
    '''
    # Create input instances
    input1 = None
    input2 = None

    # Track the four input layers
    encode1 = None
    encode2 = None
    decode1 = None
    decode2 = None

    # Number of layers per part
    encode1Num = None
    encode2Num = None
    decode1Num = None
    decode2Num = None
    embedNum = None

    # The three required los functions
    loss1 = None
    loss2 = None
    lossEmbed = None

    # Stores the compiled models
    one2one = None
    two2two = None
    one2two = None
    two2one = None
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

        self.encode1 = encodeLayers
        self.encode1Num = len(encodeLayers)
        self.decode1 = decodeLayers
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

        self.encode2 = encodeLayers
        self.encode2Num = len(encodeLayers)
        self.decode2 = decodeLayers
        self.decode2Num = len(decodeLayers)
        return 0

    def viewModel(self,encoder=0,decoder=0):
        ''' Method for viewing the models once compiled.
        encoder is either 1 or 2 for using model 1 or 2 as input.
        decoder is either 1 or 2 for using model 1 or 2 as output.
        if encoder and decoder both 0, then viewing the training model.
        '''
        try:
            if (encoder == 0) and (decoder == 0):
                self.lossModel.summary()
            elif (encoder == 1) and (decoder == 1):
                self.one2one.summary()
            elif (encoder == 1) and (decoder == 2):
                self.one2two.summary()
            elif (encoder == 2) and (decoder == 1):
                self.two2one.summary()
            elif (encoder == 2) and (decoder == 2):
                self.two2two.summary()
            else:
                print("Not a valid pair")
        except:
            print("Unable to view models until models are compiled")

    def compile(self,weights=[1,1,1]):
        # Prediction models
        embed1 = self.input1
        for count in range(self.encode1Num):
            embed1 = self.encode1[count](embed1)
        m1 = self.decode1[0](embed1)
        for count in range(1,self.decode1Num):
            m1 = self.decode1[count](m1)
        self.one2one = Model(self.input1,m1)

        embed2 = self.input2
        for count in range(self.encode2Num):
            embed2 = self.encode2[count](embed2)
        m2 = self.decode2[0](embed2)
        for count in range(1,self.decode2Num):
            m2 = self.decode2[count](m2)
        self.two2two = Model(self.input2,m2)

        # Crossing models
        m12 = self.input1
        for count in range(self.encode1Num):
            m12 = self.encode1[count](m12)
        for count in range(self.decode2Num):
            m12 = self.decode2[count](m12)
        self.one2two = Model(self.input1,m12)

        m21 = self.input2
        for count in range(self.encode2Num):
            m21 = self.encode2[count](m21)
        for count in range(self.decode1Num):
            m21 = self.decode1[count](m21)
        self.two2one = Model(self.input2,m21)

        # Training model
        self.loss1 = Lambda(lambda x:self.lossFunction(*x),name="Model1Loss")([self.input1,m1])
        self.loss2 = Lambda(lambda x:self.lossFunction(*x),name="Model2Loss")([self.input2,m2])
        self.lossEmbed = Lambda(lambda x:self.lossFunction(*x),name="CrossLoss")([embed1,embed2])
        self.lossModel = Model([self.input1,self.input2],[m1,m2,m12,m21])
        self.lossModel.add_loss(weights[0]*self.loss1)
        self.lossModel.add_loss(weights[1]*self.loss2)
        self.lossModel.add_loss(weights[2]*self.lossEmbed)
        self.lossModel.compile(optimizer='adam',loss=[None,None,None,None])

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
        pred11 = self.predict(x1,1,1)
        pred12 = self.predict(x1,1,2)
        pred21 = self.predict(x2,2,1)
        pred22 = self.predict(x2,2,2)
        error11 = mse(x1,pred11)
        error12 = mse(x2,pred12)
        error21 = mse(x1,pred21)
        error22 = mse(x2,pred22)
        return error11,error12,error21,error22

    def predict(self,x,encoder=None,decoder=None):
        if encoder is None:
            if x.shape[1:] == self.input1.shape[1:]:
                encoder = 1
            elif x.shape[1:] == self.input2.shape[1:]:
                encoder = 2
            else:
                print("Could not use either of the models.\nRemember that input should either be shape (n,{}) or (n,{})".format(self.input1.shape[1:],self.input2.shape[1:]))

        if decoder is None:
            decoder = encoder

        if (encoder == 1) and (decoder == 1):
            return self.one2one.predict(x)
        elif (encoder == 1) and (decoder == 2):
            return self.one2two.predict(x)
        elif (encoder == 2) and (decoder == 1):
            return self.two2one.predict(x)
        elif (encoder == 2) and (decoder == 2):
            return self.two2two.predict(x)
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
