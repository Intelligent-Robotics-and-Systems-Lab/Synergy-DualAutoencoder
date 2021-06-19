import pandas as pd
import numpy as np

from scipy.spatial.transform import Rotation as Rot
from sklearn.metrics import mean_squared_error as mse

def createTranform(series,quatCol,transCol):
    # Returns homogenious transform of the series[n-1]T[n]
    # quatCol must be list of 4 names referring to [q0,qx,qy,qz]
    # transCol must be list of 3 names referring to [x,y,z]
    # ex: df["Trans"] = df.apply(lambda x: createTransform(x,["q0","qx","qy","qz"],["x","y","z"]), axis=1)
    tTrans = np.zeros((4,4))
    tTrans[3][3] = 1

    tTrans[0:3][0:3] = Rot.from_quat(series[quatCol]).as_matrix()

    tTrans[0:3][3] = series[transCol]

    return transList

def createPepperTransform(df):
    # Returns the Pepper dataframe with homogenious transform [0]T[n]
    rCols = []
    lCols = []

def addStrList(string,list):
    # Adds single string at beginning of every string in the list
    return [string + "_" + l for l in list]

def createHumanDistance(df):
    # Returns the human dataframe with added columns for hand distance from core
    rHand = addStrList("RHand",["TransX","TransY","TransZ"])
    lHand = addStrList("LHand",["TransX","TransY","TransZ"])

    df["RHand_Distance"] = df.apply(lambda x: distance(x,rHand), axis=1)
    df["LHand_Distance"] = df.apply(lambda x: distance(x,lHand), axis=1)
    return df

def normalizeDistances(df):
    print(df.columns)
    # Returns the human dataframe with translation columns normalized
    tranCols = ["TransX","TransY","TransZ"]
    joints = ["LShoulder","LElbow","LHand",
        "RShoulder","RElbow","RHand"]

    def normalize(series,cols,ind):
        d = distance(series,cols)
        return series[cols[ind]] / d

    for joint in joints:
        cols = addStrList(joint,tranCols)

        df[cols[0]] = df.apply(lambda x: normalize(x,cols,0), axis=1)
        df[cols[1]] = df.apply(lambda x: normalize(x,cols,1), axis=1)
        df[cols[2]] = df.apply(lambda x: normalize(x,cols,2), axis=1)

    return df


def distance(series,cols):
    return np.sqrt(mse(np.zeros(3),series[cols].values))


def splitDistance(df,col,split):
    # Returns the df split such that the second has split percent of the largest data
    totalNum = len(df.index)
    testNum = int(np.round(totalNum * (1-split)))
    if (testNum > totalNum) or (testNum < 0):
        print("Unable to split data")
        return df, np.nan

    df = df.sort_values([col])
    return df.iloc[0:testNum], df.iloc[testNum:totalNum]


if __name__ == '__main__':
    # Running test
    file = "./Flipped/Pepper2Human/human_P01A01T01.xlsx"
    cols = ["TransX","TransY","TransZ","RotX","RotY","RotZ","RotW"]

    from readData import readPerson
    df = readPerson(file,cols)
    df = createHumanDistance(df)

    rHand = addStrList("RHand_",["TransX","TransY","TransZ"])
    print(df[rHand].iloc[0:3])
    print(df["RHand_Distance"].iloc[0:3])
