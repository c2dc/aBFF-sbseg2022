#----------------------------------------------------------------------------------------
#
#                                      train_model.py
#
#
# Input: trainDataset(${PCAP}_CIC.csv)
# Ouput: Models(${PCAP}_CIC_${ML}.joblib)
#
# Discription:
# Train models with trainDataset
#-----------------------------------------------------------------------------------------

import os
import sys
import pandas as pd
import numpy as np
import datetime
from joblib import dump
import myFunc

from sklearn.utils.random import sample_without_replacement as uSample
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler#, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, make_scorer, f1_score
from imblearn.over_sampling import ADASYN

import warnings
warnings.filterwarnings('ignore')


# GLOBAL VARIABLES 

## Select PCAP and dataset types
#
# pcapType 0: MAWILab(no attacks) + synthetic attacks
# pcapType 1: UNSW_NB15
# pcapType 2: CIC-IDS
# pcapType 3: ToN-IoT
# pcapType 4: BoT-IoT
# pcapType 5: same as 0 but with internet synthetic attacks
#
# datasetType 0: UNSW_NB15
# datasetType 1: CIC-IDS
##

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#---------#
# RUNNING #
#---------#
# Runs experiment for all algorithms on chosen dataset and saves as .joblib files
def runExperiment(pcapTypeNum, maxNumFiles, datasetTypeNum=1, scanOnly=False, scan=True, no_overwrite=True, balance="none"):
    #----------------------#
    # PREPARE FOR TRAINING #
    #----------------------#
    
    # Load training set..             using its own zero variance list
    X, y = myFunc.setTarget(myFunc.loadDataset(pcapTypeNum, maxNumFiles, datasetTypeNum), pcapTypeNum, scanOnly, scan, pcapTypeNum)

    #--------------#
    # PRE-TRAINING #
    #--------------#

    # Standard name is for models detecting scanning attacks with background classes
    filename = myFunc.getDSName(pcapTypeNum, datasetTypeNum, scanOnly, scan)
        
    #kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=17) # Train, Test
    gskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=17) # Validation
    perf = f1_score
    
    #----------#
    # TRAINING #
    #----------#
    
    #perfROC = roc_auc_score
    prep = StandardScaler() #MinMaxScaler()
    
    # Balancing
    filename = filename + myFunc.balanceNaming[balance]
    if balance == myFunc.U_SAMPLE:
        print("Undersampling majority class")
        dist = y.value_counts()
        indexMax = dist.idxmax()
        toKill = uSample(n_population=dist[indexMax], n_samples=(dist[indexMax]-dist[1-indexMax]) )
        toKillIndex = (y[y == indexMax].index)[toKill]
        y.drop(toKillIndex, inplace=True)
        X.drop(toKillIndex, inplace=True)
    if balance == myFunc.O_SAMPLE:
        print("Oversampling minority class")
        ada = ADASYN() #random_state=42)
        print('Original dataset shape {0}'.format(y))
        X, y = ada.fit_resample(X, y)
        print('Resampled dataset shape {0}'.format(y))

    # Normalize input data for training
    prep.fit(X)
    dump(prep, open('models/{0}_prep.pkl'.format(filename), 'wb'))
    for algorithm, (clf, parameters) in myFunc.algorithms.items(): #{'DT': algorithms.get('DT')}.items():
        # file path
        modelPath = "models/{0}_{1}.joblib".format(filename,algorithm)
        # if algorithm already trained and KEEP flag set
        if (os.path.isfile(modelPath)) and no_overwrite:
            print("{0} not overwriten".format(algorithm))
            continue
        #for each ML algorithm: train
        myFunc.log(filename, "training " + algorithm + " from " + filename)
        
        # F1 score
        #print("Training for F1 score")
        try:
            best = GridSearchCV(clf, parameters, cv=gskf, scoring=make_scorer(perf))
            best.fit(prep.transform(X), y)
            dump(best, modelPath)
        except Exception as e:
            myFunc.log(filename, "Something went wrong with {0} from {1}\n".format(algorithm, filename)+str(e)+"\n")

    myFunc.log(filename, "Done!")

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    
    
    
    
# IF CALLED FROM TERMINAL

if __name__ == "__main__":

    datasetMSG = "Datasets available are :\n"
    DST_MSG = "Dataset types available are :\n"
    
    pcapTypeNum = 0 # pcap file to use
    datasetTypeNum = 1 # feature set to use
    maxNumFiles = 48 # maximum number of files to load
    balanceType = "none" # what type of balancing is done
    
    #no_overwrite: skip existing joblib files, dont overwrite
    #scan: target class is Scanning\Reconnaissance
    #scanOnly: remove other attacks from data

    # help
    if len(sys.argv) < 4:
        print("Usage: " + sys.argv[0] + " <MAX_NUM_FILES> <FEATURE_SET> <PCAP_SOURCE> [\"KEEP\"] [\"SCAN_ALL\" (has precedence)] [\"SCAN_ONLY\"]")
        print(datasetMSG, myFunc.datasetOptions())
        sys.exit()
        
    if len(sys.argv) > 3:
        pcapTypeNum = int(sys.argv[3])
        datasetTypeNum = int(sys.argv[2])
        maxNumFiles = int(sys.argv[1])
        # check for unknown dataset
        if pcapTypeNum not in myFunc.pcapOptions():
            print("Unknown dataset(s): ")
            print(datasetMSG, myFunc.datasetOptions())
            sys.exit()
       
        # ToN-IoT and BoT-IoT only available in CIC dataset type
        if pcapTypeNum in [3, 4]:
            datasetTypeNum = 1
            print("ToN-IoT and BoT-IoT only available in CIC dataset type")
        # check for invalid types
        elif (datasetTypeNum not in myFunc.featureOptions()):
            print("Invalid dataset type(s): ")
            print(DST_MSG, myFunc.featureOptions())
            sys.exit()
            
    if len(sys.argv) > 4:
        if "KEEP" in sys.argv[4:]:
            no_overwrite = True
            print("No Overwrite selected. Skipping ML for existing joblib files")
        else:
            no_overwrite = False
            
        if "SCAN_ALL" in sys.argv[4:]:
            scan = True # target class is Scanning\Reconnaissance
            scanOnly = False # keep background classes
            print("Target Class: Scanning\\Reconnaissance selected")
        elif "SCAN_ONLY" in sys.argv[4:]:
            scan = True # target class is Scanning\Reconnaissance
            scanOnly = True # exclude non Scanning\Reconnaissance attacks from data
            print("Target Class: Scanning\\Reconnaissance selected, exclude other attacks from Benign data")
        else:
            scan = False # all attack classes are targeted
            scanOnly = False # keep background classes
            
        if myFunc.U_SAMPLE in sys.argv[4:]:
            balanceType = myFunc.U_SAMPLE
            
        if myFunc.O_SAMPLE in sys.argv[4:]:
            balanceType = myFunc.O_SAMPLE
        
    runExperiment(pcapTypeNum, maxNumFiles, datasetTypeNum, scanOnly, scan, no_overwrite, balance=balanceType)
    
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX