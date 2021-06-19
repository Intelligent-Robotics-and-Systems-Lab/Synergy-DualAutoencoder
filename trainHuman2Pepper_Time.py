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

def createSets(colsH,humanID):
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

    humanDF,humanVal,pepperDF,pepperVal = train_test_split(humanDF,pepperDF,test_size=0.12)
    humanTrain,humanTest,pepperTrain,pepperTest = train_test_split(humanDF,pepperDF,test_size=0.12)
    '''
    humanDF = createHumanDistance(humanDF)
    humanDF,humanVal = splitDistance(humanDF,choice(["RHand_Distance","LHand_Distance"]),0.12)
    humanTrain,humanTest = splitDistance(humanDF,choice(["RHand_Distance","LHand_Distance"]),0.12)

    print(humanTrain.columns)
    humanTrain = humanTrain.drop(["RHand_Distance","LHand_Distance"],axis=1)
    humanTest = humanTest.drop(["RHand_Distance","LHand_Distance"],axis=1)
    humanVal = humanVal.drop(["RHand_Distance","LHand_Distance"],axis=1)

    pepperVal = pepperDF.loc[humanVal.index]
    pepperTest = pepperDF.loc[humanTest.index]
    pepperTrain = pepperDF.loc[humanTrain.index]
    '''
    return humanTrain,humanVal,humanTest,pepperTrain,pepperVal,pepperTest

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

def plotHistoryAll(history,xlim=None,ylim=None,showLegend=True):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    names = ("Direct","ReDa","CycleGAN")
    for h,n in zip(history,names):
        ax.plot(h.history['loss'],label=n+" Train")
        ax.plot(h.history['val_loss'],label=n+" Valid")

    ax.set_title("Model Validation")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if showLegend:
        ax.legend(loc="upper right")
    return fig


if __name__ == '__main__':
    fileName = "./results/samples.xlsx" # None if file has not been created yet
    humanID = 2
    epoch=200
    batchSize=20
    verbose=0
    repeats = 15

    # Create Models
    logging.info("Create Networks:")
    enHSize = [16,14,12,10]
    deHSize = [12,14,16,18]
    enPSize = [32,24,16,10]
    dePSize = [16,24,32,40]
    inHShape = 18
    inPShape = 40

    colsH = ["TransX","TransY","TransZ"]

    # Train models
    timeDHP = []
    timeDPH = []
    timeST = []
    timeGAN = []
    mseDHP = []
    mseDPH = []
    mseSTHP = []
    mseSTPH = []
    mseGANHP = []
    mseGANPH = []
    for count in range(repeats):
        humanTrain,humanVal,humanTest,pepperTrain,pepperVal,pepperTest = createSets(colsH,humanID)

        histories = []
        stModel = createSindy(enHSize,deHSize,enPSize,dePSize,inHShape,inPShape)
        GANModel = createGAN(enHSize,deHSize,enPSize,dePSize,inHShape,inPShape)
        directAutoHP,directAutoPH = createDirect([17,16,15,14,13,12,11,10],[11,12,13,14,15,16,17,18],
            [36,32,28,24,20,16,13,10],[13,16,20,24,28,32,36,40],
            inHShape,inPShape)

        startTime = time.time()
        logging.info("Training Direct Model Human to Pepper")
        history = directAutoHP.fit(humanTrain,pepperTrain,validation_data=(humanVal,pepperVal),epochs=epoch,batch_size=batchSize,verbose=verbose)
        histories.append(history)

        fig = plotHistory(history)
        try:
            fig.savefig("./results/directHuman2Pepper/validation{}.png".format(count))
        except:
            mkdir("./results/directHuman2Pepper")
            fig.savefig("./results/directHuman2Pepper/validation{}.png".format(count))
        pred = directAutoHP.predict(humanTest)
        mseDHP.append(np.sqrt(mse(pred,pepperTest)))
        logging.info("RMSE for direct human to pepper mapping is {}".format(mseDHP[-1]))
        pred = pd.DataFrame(pred,columns=pepperTest.columns)
        savePepper("./results/directHuman2Pepper/human2pepper{}".format(count),pred)
        timeDHP.append(time.time() - startTime)
        logging.info("Ran First Part in {} seconds.".format(timeDHP[-1]))

        startTime = time.time()
        logging.info("Training Direct Model Pepper to Human")
        history = directAutoPH.fit(pepperTrain,humanTrain,validation_data=(pepperVal,humanVal),epochs=epoch,batch_size=batchSize,verbose=verbose)

        fig = plotHistory(history)
        try:
            fig.savefig("./results/directPepper2Human/validation{}.png".format(count))
        except:
            mkdir("./results/directPepper2Human")
            fig.savefig("./results/directPepper2Human/validation{}.png".format(count))
        pred = directAutoPH.predict(pepperTest)
        mseDPH.append(np.sqrt(mse(pred,humanTest)))
        logging.info("RMSE for direct pepper to human mapping is {}".format(mseDPH[-1]))
        pred = pd.DataFrame(pred,columns=humanTest.columns)
        saveHuman("./results/directPepper2Human/pepper2human{}".format(count),pred)
        timeDPH.append(time.time() - startTime)
        logging.info("Ran Second Part in {} seconds.".format(timeDPH[-1]))
        logging.info("Ran Direct in {} seconds.".format(timeDPH[-1] + timeDHP[-1]))

        startTime = time.time()
        logging.info("Training Sindy-DualAutoencoder Model")
        history = stModel.fit(humanTrain,pepperTrain,val1=humanVal,val2=pepperVal,epochs=epoch,batch_size=batchSize,verbose=verbose)
        histories.append(history)

        fig = plotHistory(history)
        try:
            fig.savefig("./results/SindyDualAutoencoder/validation{}.png".format(count))
        except:
            mkdir("./results/SindyDualAutoencoder")
            fig.savefig("./results/SindyDualAutoencoder/validation{}.png".format(count))
        pred = stModel.predict(humanTest,1,2)
        mseSTHP.append(np.sqrt(mse(pred,pepperTest)))
        logging.info("RMSE for direct human to pepper mapping is {}".format(mseSTHP[-1]))
        pred = pd.DataFrame(pred,columns=pepperTest.columns)
        savePepper("./results/SindyDualAutoencoder/human2pepper{}".format(count),pred)
        pred = stModel.predict(pepperTest,2,1)
        mseSTPH.append(np.sqrt(mse(pred,humanTest)))
        logging.info("RMSE for direct pepper to human mapping is {}".format(mseSTPH[-1]))
        pred = pd.DataFrame(pred,columns=humanTest.columns)
        saveHuman("./results/SindyDualAutoencoder/pepper2human{}".format(count),pred)
        timeST.append(time.time() - startTime)
        logging.info("Ran Sindy-DualAutoencoder in {} seconds.".format(timeST[-1]))

        startTime = time.time()
        logging.info("Training cycleGAN Model")
        history = GANModel.fit(humanTrain,pepperTrain,val1=humanVal,val2=pepperVal,epochs=epoch,batch_size=batchSize,verbose=verbose)
        histories.append(history)

        fig = plotHistory(history)
        try:
            fig.savefig("./results/cycleGAN/validation{}.png".format(count))
        except:
            mkdir("./results/cycleGAN")
            fig.savefig("./results/cycleGAN/validation{}.png".format(count))
        pred = GANModel.predict(humanTest,1,2)
        mseGANHP.append(np.sqrt(mse(pred,pepperTest)))
        logging.info("RMSE for direct human to pepper mapping is {}".format(mseGANHP[-1]))
        pred = pd.DataFrame(pred,columns=pepperTest.columns)
        savePepper("./results/cycleGAN/human2pepper{}".format(count),pred)
        pred = GANModel.predict(pepperTest,2,1)
        mseGANPH.append(np.sqrt(mse(pred,humanTest)))
        logging.info("RMSE for direct pepper to human mapping is {}".format(mseGANPH[-1]))
        pred = pd.DataFrame(pred,columns=humanTest.columns)
        saveHuman("./results/cycleGAN/pepper2human{}".format(count),pred)
        timeGAN.append(time.time() - startTime)
        logging.info("Ran cycleGAN in {} seconds.".format(timeGAN[-1]))

        fig = plotHistoryAll(histories)
        fig.savefig("./results/validation{}.png".format(count))
        fig = plotHistoryAll(histories,(0,50),(0,0.1),False)
        fig.savefig("./results/validation{}_sub.png".format(count))

    saveHuman("./results/realhuman",humanTest)
    savePepper("./results/realpepper",pepperTest)

    #Average Results
    timeDHP = np.average(timeDHP)
    timeDPH = np.average(timeDPH)
    timeST = np.average(timeST)
    timeGAN = np.average(timeGAN)

    stdDHP = np.std(mseDHP)
    stdDPH = np.std(mseDPH)
    stdSTHP = np.std(mseSTHP)
    stdSTPH = np.std(mseSTPH)
    stdGANHP = np.std(mseGANHP)
    stdGANPH = np.std(mseGANPH)

    mseDHP = np.average(mseDHP)
    mseDPH = np.average(mseDPH)
    mseSTHP = np.average(mseSTHP)
    mseSTPH = np.average(mseSTPH)
    mseGANHP = np.average(mseGANHP)
    mseGANPH = np.average(mseGANPH)

    logging.info("Total Results")
    logging.info("Ran part one of direct autoencoder (human to pepper) in {} seconds with rmse of {}, {}".format(timeDHP,mseDHP,stdDHP))
    logging.info("Ran part two of direct autoencoder (pepper to human) in {} seconds with rmse of {}, {}".format(timeDPH,mseDPH,stdDPH))
    logging.info("Trained full direct autoencoder in {} seconds".format(timeDHP+timeDPH))
    logging.info("Ran Sindy-DualAutoencoder in human to pepper mode with rmse of {}, {}".format(mseSTHP,stdSTHP))
    logging.info("Ran Sindy-DualAutoencoder in pepper to human mode with rmse of {}, {}".format(mseSTPH,stdSTPH))
    logging.info("Trained Sindy-DualAutoencoder in {} seconds".format(timeST))
    logging.info("Ran cycleGAN in human to pepper mode with rmse of {}, {}".format(mseGANHP,stdGANHP))
    logging.info("Ran cycleGAN in pepper to human mode with rmse of {}, {}".format(mseGANPH,stdGANPH))
    logging.info("Trained cycleGAN in {} seconds".format(timeGAN))

    logging.info("Train Shape {}".format(humanTrain.shape))
    logging.info("Val Shape {}".format(humanVal.shape))
    logging.info("Test Shape {}".format(humanTest.shape))
