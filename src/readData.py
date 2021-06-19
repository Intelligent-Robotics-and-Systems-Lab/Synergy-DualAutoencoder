import pandas as pd
import numpy as np
import glob

def readExcel(file,keepCol,joints):
    '''Reads the excel sheet and format data for machine learning
    file is the name of the excel sheet
    keepCol is the array of columns to use in the excel sheet
    joints is the array of sheet names in the excel sheet

    Returns a dataframe with rows equal to length of keepCol times length of joints and column equal to rows in excel sheet - 2
    '''

    # Grab all data from the first joint in the right format
    tempDF = pd.read_excel(file,sheet_name=joints[0],index_col=None)
    tempDF = renameCol(tempDF[keepCol],joints[0])
    data = np.transpose(tempDF)

    # Grab and format the remaining joints
    for i in range(1,len(joints)):
        tempDF = pd.read_excel(file,sheet_name=joints[i],index_col=None)
        tempDF = renameCol(tempDF[keepCol],joints[i])
        data = data.append(np.transpose(tempDF))
    return np.transpose(data)

def readPerson(file,cols):
    '''Calls readExcel with only human points
    '''
    '''joints = ["Head","Neck","Torso",
        "LShoulder","LElbow","LHand",
        "RShoulder","RElbow","RHand",
        "LHip","RHip"] # Do not have Head, Knees, or Feet'''
    joints = ["LShoulder","LElbow","LHand",
        "RShoulder","RElbow","RHand"]
    xl = pd.ExcelFile(file)
    jts = []
    for jt in joints:
        if jt in xl.sheet_names:
            jts.append(jt)
    return readExcel(file,cols,jts)

def readPepper(file,cols):
    '''Calls readExcel with only pepper points
    '''
    '''joints = ["Head","Neck","Torso",
        "LShoulder","LBicep","LElbow","LForeArm","LHand",
        "RShoulder","RBicep","RElbow","RForeArm","RHand",
        "Hip","Pelvis","Tibia"] # Do not have Head, Knees, Feet, or wheels'''
    joints = ["LShoulder","LBicep","LElbow","LForeArm","LHand",
        "RShoulder","RBicep","RElbow","RForeArm","RHand"]
    return readExcel(file,cols,joints)

def readData(folderHuman,folderPepper,colHuman,colPepper,humanID,pepperID):
    '''Reads all file names in the two folders to find human and pepper matches
    folderHuman is the name of the folder which contains all human trials
    folderPepper is the name of the folder which contains all pepper trials
    cols are the columns to use in every excel sheet

    Returns the human and pepper data in respective dataframes
    '''
    print(folderHuman+"/*P{:02d}*.xlsx".format(humanID))
    humanFiles = sorted(glob.glob(folderHuman+"/*P{:02d}*.xlsx".format(humanID)))
    print(humanFiles)

    # Find pepper file that matches each action from the human
    humanDF = pd.DataFrame()
    pepperDF = pd.DataFrame()
    for hf in humanFiles:
        p,a,t = extractTrialInfo(hf)
        pf = folderPepper+"/pepper_P{:02d}A".format(pepperID)+a+"T01.xlsx"
        try:
            pepperDF = mergeData(pepperDF,readPepper(pf,colPepper))
            humanDF = mergeData(humanDF,readPerson(hf,colHuman))
        except:
            print("Unable to work for these files:")
            print(hf)
            print(pf)
            print(dfadaf)
    return humanDF,pepperDF

def extractTrialInfo(file):
    '''For a file name in form *_PxxAyyTzz.xlsx,
    returns the person (xx), action (yy), and trial (zz) as ints.
    '''
    trial = file[-7:-5]
    action = file[-10:-8]
    person = file[-13:-11]
    return person,action,trial

def renameCol(df,name):
    '''Adds a string in from of existing column names
    '''
    columns = {}
    for i in range(len(df.columns)):
        columns[df.columns[i]] = name + "_" + df.columns[i]
    return df.rename(columns=columns)

def mergeData(df1,df2):
    '''Used to merge two sets of data from the same source
    '''
    if df1.empty:
        return df2
    elif df2.empty:
        return df1
    else:
        return df1.append(df2,ignore_index=True)

def saveHuman(folder,data):
    fileName = folder + "Trial.xlsx"
    joints = ["Head","Neck","Torso",
        "LShoulder","LElbow","LHand",
        "RShoulder","RElbow","RHand",
        "LHip","LKnee","LFoot",
        "RHip","RKnee","RFoot"]

    with pd.ExcelWriter(fileName) as writer:
        for joint in joints:
            df = data.filter(regex=joint)
            if not df.empty:
                df.columns = df.columns.str.split("_").str[1]
                df.to_excel(writer,sheet_name=joint,index=False)

def savePepper(folder,data):
    fileName = folder + "Trial.xlsx"
    joints = ["Head","Neck","Torso",
        "LShoulder","LBicep","LElbow","LForeArm","LHand",
        "RShoulder","RBicep","RElbow","RForeArm","RHand",
        "Hip","Pelvis","Tibia",
        "WheelB_link","RKnee","RFoot"]
    posses = [[0,0,0],[-0.038,0,0.1699],[0,0,0],
        [-0.057,0.14974,0.08682],[0,0,0],[0.1812,0.015,0.00013],[0,0,0],[0.15,0,0],
        [-0.057,-0.14974,0.08682],[0,0,0],[0.1812,-0.015,0.00013],[0,0,0],[0.15,0,0],
        [0.00002,0,-0.139],[0,0,-0.079],[0,0,-0.268],
        [-0.17,0,-0.264],[0.09,0.155,-0.264],[0.09,-0.155,-0.264]]

    with pd.ExcelWriter(fileName) as writer:
        for joint,pos in zip(joints,posses):
            df = data.filter(regex=joint)
            if not df.empty:
                df.columns = df.columns.str.split("_").str[1]
                df["TransX"] = pos[0]
                df["TransY"] = pos[1]
                df["TransZ"] = pos[2]
                df.to_excel(writer,sheet_name=joint,index=False)

if __name__ == '__main__':
    colsP = ["RotX","RotY","RotZ","RotW"]
    colsH = ["TransX","TransY","TransZ"]
    humans = "./humanData"
    peppers = "./pepperData"

    humanDF,pepperDF = readData(humans,peppers,colsH,colsP)
    print(humanDF)
    print(pepperDF)
