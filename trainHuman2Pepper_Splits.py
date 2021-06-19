from src.readData import *
from src.CycleGAN import *
from src.SindyDualAutoencoder import *
from src.dynamics import *

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, LeakyReLU as lrelu
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import mkdir
from random import choice
import time

import logging
try:
    logging.basicConfig(filename="./results/output.txt",level=logging.INFO)
except:
    mkdir("./results/")
    logging.basicConfig(filename="./results/output.txt",level=logging.INFO)

def createSets(colsH,humanID,split):
    # Important features and location of data
    colsP = ["RotX","RotY","RotZ","RotW"]
    humanFolder = "./DataSets/Flipped/Pepper2Human"
    pepperFolder = "./DataSets/Flipped/Pepper"
    pepperID = 1

    # Read data and get count
    humanDF,pepperDF=readData(humanFolder,pepperFolder,colsH,colsP,humanID,pepperID)
    humanDF = normalizeDistances(humanDF)
    hSamples,hFeatures = humanDF.shape
    pSamples,pFeatures = pepperDF.shape

    # Split into training and testing data
    humanTrain,humanTest,pepperTrain,pepperTest = train_test_split(humanDF,pepperDF,test_size=split)
    '''
    humanDF = createHumanDistance(humanDF)
    humanTrain,humanTest = splitDistance(humanDF,choice(["RHand_Distance","LHand_Distance"]),0.12)
    humanTrain = humanTrain.drop(["RHand_Distance","LHand_Distance"],axis=1)
    humanTest = humanTest.drop(["RHand_Distance","LHand_Distance"],axis=1)
    pepperTest = pepperDF.loc[humanTest.index]
    pepperTrain = pepperDF.loc[humanTrain.index]
    '''

    return humanTrain,humanTest,pepperTrain,pepperTest

def createSindy(en1Size,de1Size,en2Size,de2Size,in1Shape,in2Shape):
    #Sindy-DualAutoencoder
    stModel = SindyDualAutoencoderModel(in1Shape,in2Shape)

    enLayer = []
    deLayer = []
    for count in range(len(en1Size)):
        enLayer.append(Dense(en1Size[count],activation=lrelu(alpha=0.01)))
    for count in range(len(de1Size)-1):
        deLayer.append(Dense(de1Size[count],activation=lrelu(alpha=0.01)))
    deLayer.append(Dense(de1Size[-1]))
    stModel.createModel1(enLayer,deLayer)

    enLayer = []
    deLayer = []
    for count in range(len(en2Size)):
        enLayer.append(Dense(en2Size[count],activation=lrelu(alpha=0.01)))
    for count in range(len(de2Size)-1):
        deLayer.append(Dense(de2Size[count],activation=lrelu(alpha=0.01)))
    deLayer.append(Dense(de2Size[-1]))
    stModel.createModel2(enLayer,deLayer)

    stModel.compile()
    return stModel

def createGAN(en1Size,de1Size,en2Size,de2Size,in1Shape,in2Shape):
    #cycleGAN
    GANModel = CycleGANModel(in1Shape,in2Shape)

    enLayer = []
    deLayer = []
    for count in range(len(en1Size)):
        enLayer.append(Dense(en1Size[count],activation=lrelu(alpha=0.01)))
    for count in range(len(de1Size)-1):
        deLayer.append(Dense(de1Size[count],activation=lrelu(alpha=0.01)))
    deLayer.append(Dense(de1Size[-1]))
    GANModel.createModel1(enLayer,deLayer)

    enLayer = []
    deLayer = []
    for count in range(len(en2Size)):
        enLayer.append(Dense(en2Size[count],activation=lrelu(alpha=0.01)))
    for count in range(len(de2Size)-1):
        deLayer.append(Dense(de2Size[count],activation=lrelu(alpha=0.01)))
    deLayer.append(Dense(de2Size[-1]))
    GANModel.createModel2(enLayer,deLayer)

    GANModel.compile()
    return GANModel

def createDirect(en1Size,de1Size,en2Size,de2Size,in1Shape,in2Shape):
    #Traditional Autoencoder
    da1in = Input(shape=in1Shape)
    da1 = Dense(en1Size[0],activation=lrelu(alpha=0.01))(da1in)
    for count in range(1,len(en1Size)):
        da1 = Dense(en1Size[count],activation=lrelu(alpha=0.01))(da1)
    for count in range(len(de2Size)-1):
        da1 = Dense(de2Size[count],activation=lrelu(alpha=0.01))(da1)
    da1 = Dense(de2Size[-1])(da1)

    da2in = Input(shape=in2Shape)
    da2 = Dense(en2Size[0],activation=lrelu(alpha=0.01))(da2in)
    for count in range(len(en2Size)):
        da2 = Dense(en2Size[count],activation=lrelu(alpha=0.01))(da2)
    for count in range(len(de1Size)-1):
        da2 = Dense(de1Size[count],activation=lrelu(alpha=0.01))(da2)
    da2 = Dense(de1Size[-1])(da2)

    loss = Huber(delta=0.6)

    directAuto12 = Model(da1in,da1)
    directAuto12.compile(optimizer="adam",loss=loss)
    directAuto21 = Model(da2in,da2)
    directAuto21.compile(optimizer="adam",loss=loss)
    return directAuto12,directAuto21

def plotHistory(history):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title("Validation Loss")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    return fig

def plotHistoryAll(history):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    names = ("Direct","ReDa","CycleGAN")
    for h,n in zip(history,names):
        ax.plot(h.history['loss'],label=n+" Train")
        ax.plot(h.history['val_loss'],label=n+" Valid")

    ax.set_title("Model Validation")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc="upper right")
    return fig


if __name__ == '__main__':
    fileName = "./results/samples.xlsx" # None if file has not been created yet
    humanID = 2
    epoch=200
    batchSize=20
    verbose=0
    repeats = 40
    splits = [.1,.15,.2,.25,.3]

    # Create Models
    logging.info("Create Networks:")
    enHSize = [16,14,12,10]
    deHSize = [12,14,16,18]
    enPSize = [32,24,16,10]
    dePSize = [16,24,32,40]
    inHShape = 18
    inPShape = 40

    colsH = ["TransX","TransY","TransZ"]

    mseDHPTot = []
    mseDPHTot = []
    mseSTHPTot = []
    mseSTPHTot = []
    mseGANHPTot = []
    mseGANPHTot = []
    stdDHPTot = []
    stdDPHTot = []
    stdSTHPTot = []
    stdSTPHTot = []
    stdGANHPTot = []
    stdGANPHTot = []
    for split in splits:
        logging.info("Splitting train and test for 15% testing")

        # Train models
        mseDHP = []
        mseDPH = []
        mseSTHP = []
        mseSTPH = []
        mseGANHP = []
        mseGANPH = []
        stdDHP = []
        stdDPH = []
        stdSTHP = []
        stdSTPH = []
        stdGANHP = []
        stdGANPH = []
        for count in range(repeats):
            humanTrain,humanTest,pepperTrain,pepperTest = createSets(colsH,humanID,split)

            histories = []
            stModel = createSindy(enHSize,deHSize,enPSize,dePSize,inHShape,inPShape)
            GANModel = createGAN(enHSize,deHSize,enPSize,dePSize,inHShape,inPShape)
            directAutoHP,directAutoPH = createDirect([17,16,15,14,13,12,11,10],[11,12,13,14,15,16,17,18],
                [36,32,28,24,20,16,13,10],[13,16,20,24,28,32,36,40],
                inHShape,inPShape)

            logging.info("Training Direct Model Human to Pepper")
            directAutoHP.fit(humanTrain,pepperTrain,epochs=epoch,batch_size=batchSize,verbose=verbose)

            pred = directAutoHP.predict(humanTest)
            mseDHP.append(np.sqrt(mse(pred,pepperTest)))
            logging.info("RMSE for direct human to pepper mapping is {}".format(mseDHP[-1]))


            logging.info("Training Direct Model Pepper to Human")
            directAutoPH.fit(pepperTrain,humanTrain,epochs=epoch,batch_size=batchSize,verbose=verbose)

            pred = directAutoPH.predict(pepperTest)
            mseDPH.append(np.sqrt(mse(pred,humanTest)))
            logging.info("RMSE for direct pepper to human mapping is {}".format(mseDPH[-1]))


            logging.info("Training Sindy-DualAutoencoder Model")
            stModel.fit(humanTrain,pepperTrain,epochs=epoch,batch_size=batchSize,verbose=verbose)

            pred = stModel.predict(humanTest,1,2)
            mseSTHP.append(np.sqrt(mse(pred,pepperTest)))
            logging.info("RMSE for direct human to pepper mapping is {}".format(mseSTHP[-1]))
            pred = stModel.predict(pepperTest,2,1)
            mseSTPH.append(np.sqrt(mse(pred,humanTest)))
            logging.info("RMSE for direct pepper to human mapping is {}".format(mseSTPH[-1]))


            logging.info("Training cycleGAN Model")
            GANModel.fit(humanTrain,pepperTrain,epochs=epoch,batch_size=batchSize,verbose=verbose)

            pred = GANModel.predict(humanTest,1,2)
            mseGANHP.append(np.sqrt(mse(pred,pepperTest)))
            logging.info("RMSE for direct human to pepper mapping is {}".format(mseGANHP[-1]))
            pred = GANModel.predict(pepperTest,2,1)
            mseGANPH.append(np.sqrt(mse(pred,humanTest)))
            logging.info("RMSE for direct pepper to human mapping is {}".format(mseGANPH[-1]))


        #Average Results
        mseDHPTot.append(np.average(mseDHP))
        mseDPHTot.append(np.average(mseDPH))
        mseSTHPTot.append(np.average(mseSTHP))
        mseSTPHTot.append(np.average(mseSTPH))
        mseGANHPTot.append(np.average(mseGANHP))
        mseGANPHTot.append(np.average(mseGANPH))
        stdDHPTot.append(np.std(mseDHP))
        stdDPHTot.append(np.std(mseDPH))
        stdSTHPTot.append(np.std(mseSTHP))
        stdSTPHTot.append(np.std(mseSTPH))
        stdGANHPTot.append(np.std(mseGANHP))
        stdGANPHTot.append(np.std(mseGANPH))

    df = pd.DataFrame({"mseDHP":mseDHPTot,"stdDHP":stdDHPTot, "mseDPH":mseDPHTot,"stdDPH":stdDPHTot,
                       "mseSTHP":mseSTHPTot,"stdSTHP":stdSTHPTot, "mseSTPH":mseSTPHTot,"stdSTPH":stdSTPHTot,
                       "mseGANHP":mseGANHPTot,"stdGANHP":stdGANHPTot, "mseGANPH":mseGANPHTot,"stdGANPH":stdGANPHTot})
    df.to_csv("./results/data.csv")
