#----------------------------------------------------------------------------------------
#
#                                      trainTestCIC.py
#
#
# Input: trainDataset(${PCAP}_CIC.csv) testDataset(${PCAP}_CIC.csv)[list]
# Ouput: (${PCAP}_CIC.csv)
#
# Discription:
# Train with trainDataset and test with testDataset list
#-----------------------------------------------------------------------------------------

import os
import sys
import pandas as pd
import numpy as np
from joblib import dump, load
import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

#-----------#
# FUNCTIONS #
#-----------#

# def zeroVarWrite(ZV,pcapTypeNum)
# def zeroVarRead(pcapTypeNum)

# def getFilename(pcapTypeNum, datasetTypeNum)
# def getDSName(pNum, dNum=1, scanOnly=False, scan=True)

# def log(DSName, info)
# def pcapOptions()
# def featureOptions()
# def datasetOptions()
# def saveTable(table, tableName, caption, label)

# def standardCICFeatureName(features)
# def tempoFunc(x)
# def sickCICFeatureName(test)
# def buildComparisonTable(scanOnly, scan)

# def loadModel(modelType)
# def loadDataset(pNum, maxNumFiles, dNum, filepath="./dataset/", BUILD=False)
# def buildDataset(pcapTypeNum, maxNumFiles, datasetTypeNum, filepath)
# def setTarget(full_data, pNum, scanOnly, scan, zeroVarType)

#------------------#
# GLOBAL VARIABLES #
#------------------#

## Select PCAP and dataset types
#
# pcapType 0: MAWILab(no attacks) + synthetic attacks
# pcapType 1: UNSW_NB15
# pcapType 2: CIC-IDS
# pcapType 3: ToN-IoT
# pcapType 4: BoT-IoT
#
# datasetType 0: UNSW_NB15
# datasetType 1: CIC-IDS
##

# Dictionary of types to filename affixes used in file selection
pcapType = {0:"output", 1:"NB15_", 2:"WorkingHours", 3:"ToN-IoT", 4:"BoT-IoT", 5:"internet"}
datasetType = {0:"_NB15.csv", 1:"_CIC.csv"}

# Define ML algorithms
ALGORITHMS = {
    "MLP" : (MLPClassifier(random_state=17), {
        "hidden_layer_sizes" : (10, 10),
    }),
    "SVM" : (LinearSVC(random_state=17), {}),
    "KNN" : (KNeighborsClassifier(n_jobs=-1), {
        "n_neighbors" : [1, 3, 5]
    }),
    "XGB" : (XGBClassifier(random_state=17, n_jobs=-1,verbosity=0), {}),
    "NB" : (GaussianNB(), {}),
    "LR" : (LogisticRegression(random_state=17, n_jobs=-1), {}),
    "RF" : (RandomForestClassifier(random_state=17, n_jobs=-1), {
        "n_estimators" : [10, 15, 20],
        "criterion" : ("gini", "entropy"), 
        "max_depth": [5, 10],
        "class_weight": (None, "balanced", "balanced_subsample")
    }),
    "DT" : (DecisionTreeClassifier(random_state=17), {
        "criterion": ("gini", "entropy"), 
        "max_depth": [5, 10, 15],
        "class_weight": (None, "balanced")
    }),
}
ALGO_KEYS = ["MLP", "SVM", "XGB", "NB", "LR", "DT"]
algorithms = {key: ALGORITHMS[key] for key in ALGO_KEYS}

FEATURE_TYPES = {'flow_duration':'float32', 'flow_byts_s':'float32', 'flow_pkts_s':'float32', 'fwd_pkts_s':'float32', 'bwd_pkts_s':'float32',
                 'fwd_pkt_len_max':'int32', 'fwd_pkt_len_min':'int32', 'bwd_pkt_len_max':'int32', 'bwd_pkt_len_min':'int32',
                 'flow_iat_max':'float32','flow_iat_min':'float32', 'fwd_iat_tot':'float32', 'fwd_iat_max':'float32', 'fwd_iat_min':'float32', 
                 'bwd_iat_tot':'int32', 'bwd_iat_max':'int32','bwd_iat_min':'int32', 'active_max':'float32', 'active_min':'float32',
                 'idle_max':'int32', 'idle_min':'int32', 'protocol':'int32','tot_fwd_pkts':'int32','tot_bwd_pkts':'int32', 
                 'totlen_fwd_pkts':'int32', 'totlen_bwd_pkts':'int32','pkt_len_max':'int32', 'pkt_len_min':'int32','fwd_header_len':'int32', 
                 'bwd_header_len':'int32','fwd_seg_size_min':'int32', 'fwd_act_data_pkts':'int32', 'fwd_psh_flags':'int32', 
                 'bwd_psh_flags':'int32','fwd_urg_flags':'int32','bwd_urg_flags':'int32', 'fin_flag_cnt':'int32',
                  'syn_flag_cnt':'int32', 'rst_flag_cnt':'int32','psh_flag_cnt':'int32', 'ack_flag_cnt':'int32','urg_flag_cnt':'int32',
                  'ece_flag_cnt':'int32','init_fwd_win_byts':'int32','init_bwd_win_byts':'int32', 'cwe_flag_count':'int32', 
                 'subflow_fwd_pkts':'int32','subflow_bwd_pkts':'int32','subflow_fwd_byts':'int32', 'subflow_bwd_byts':'int32', 
                 'src_port':'int32','dst_port':'int32','fwd_pkt_len_mean':'float32','fwd_pkt_len_std':'float32', 'bwd_pkt_len_mean':'float32',
                 'bwd_pkt_len_std':'float32', 'pkt_len_mean':'float32', 
                  'pkt_len_std':'float32','pkt_len_var':'float32','flow_iat_mean':'float32','flow_iat_std':'float32', 'fwd_iat_mean':'float32',
                  'fwd_iat_std':'float32','bwd_iat_mean':'float32','bwd_iat_std':'float32','down_up_ratio':'float32', 'pkt_size_avg':'float32', 
                  'active_mean':'float32','active_std':'float32','idle_mean':'float32','idle_std':'float32', 'fwd_byts_b_avg':'float32',
                  'fwd_pkts_b_avg':'float32', 'bwd_byts_b_avg':'float32','bwd_pkts_b_avg':'float32','fwd_blk_rate_avg':'float32', 
                 'bwd_blk_rate_avg':'float32','fwd_seg_size_avg':'float32','bwd_seg_size_avg':'float32','src_ip':'string', 'dst_ip':'string',
                 'timestamp':'string', 'flow_ID':'string', 'Label':'string'}

ID_FEATURES = ['timestamp','flow_ID', 'src_port', 'src_ip', 'dst_ip'] # removed with the zero variance features
ALL_ID = ['timestamp','flow_ID', 'src_port', 'src_ip', 'dst_ip', 'dst_port']

alter = {0:"AB-TRAP", 1:"NB15", 2:"CIC-IDS"} # used in file naming control
scatag = "SCAN_"                             # used in file naming control
atktag = "ATK_"                              # used in file naming control

U_SAMPLE = "undersampling"
O_SAMPLE = "oversampling"
balanceNaming = {"none":"", U_SAMPLE:"_us", O_SAMPLE:"_os"} # used in file naming control

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#--------------------------#
# ZERO VARIANCE READ/WRITE #
#--------------------------#

## Write zero variance feature names into text file: comma separated, no spaces
# Uses global pcapTypeNum value

# TO_DO: Add misssing dNum, scanOnly and scan variables for other feature sets and attack classes
def zeroVarWrite(ZV,pcapTypeNum):
    name = "zeroVar{0}.txt".format(getDSName(pcapTypeNum))
    print("writing file: ".format(name))
    
    featFile = open("./ML-output/{0}".format(name),"w")
    featFile.write(",".join(ZV))
    featFile.close()
    
## Read zero variance feature names from text file: comma separated, no spaces
def zeroVarRead(pcapTypeNum):
    name = "zeroVar{0}.txt".format(getDSName(pcapTypeNum))
    print("reading file: ".format(name))
    
    featFile = open("./ML-output/{0}".format(name),"r")
    ZV = featFile.read()
    featFile.close()
    ZV = ZV.split(",")
    return ZV


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#---------------------#
# FILE NAMING CONTROL #
#---------------------#

## Get file name from pcap source and feature set type numbers. Used while using fragmented files
def getFilename(pcapTypeNum, datasetTypeNum):
    return pcapType[pcapTypeNum] + datasetType[datasetTypeNum].replace(".csv","")

## Get data set name from pcap source and feature set type numbers
def getDSName(pNum, dNum=1, scanOnly=False, scan=True):
    name = pcapType[pNum]
    if pNum in alter.keys():
        name = alter[pNum]
    if scanOnly:
        # SCAN_ models learned only from scanning attacks
        name = scatag+name
    elif not scan and pNum:
        # ATK_ models detect attacks as a single class
        name = atktag+name
    return name+datasetType[dNum].replace(".csv","")


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#------------------#
# HELPER FUNCTIONS #
#------------------#

## Log data set information
def log(DSName, data):
    info = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"\n"+data+"\n"
    logFile = open("./dissertation/log_{0}.txt".format(DSName),"a")
    logFile.write(info)
    logFile.close()
    print(info)

## Get pcap number options
def pcapOptions():
    return pcapType.keys()

## Get feature set number options
def featureOptions():
    return datasetType.keys()

## Get data set options
def datasetOptions():
    setNames = pcapType.copy()
    setNames.update(alter)
    return setNames
    #setNames = setNames.values
    #return {x:setNames[x] for x in range( len(setNames) )}

## Save LaTex format table of models performance for a given training dataset [update to different tables on same function]
def saveTable(table, tableName, mycaption, mylabel):
    featFile = open("./dissertation/{0}.tex".format(tableName),"w")
    featFile.write(table.to_latex(column_format='c'*table.columns.size, index=False,
                                  caption=mycaption, label=mylabel, position="H")
                  )
    featFile.close()

    
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#--------------#
# LABEL FIXERS #
#--------------#

## fix for ToN-IoT and BoT-IoT and rogue white spaces
def standardCICFeatureName(features):
    columns = features.to_series().apply(lambda x: x.strip().replace(" ","_").replace("/","_").casefold())
    columns[columns == "flow_id"] = 'flow_ID'
    columns[columns == "label"] = 'Label'
    return columns

## fix for massive inconsistencies in CIC-IDS2017 feature set
# helper for sickCICFeatureName()
def tempoFunc(x):
    if x.find("seg_size") > 0:
        x = x.replace("_seg_size", "")+"_seg_size"
    if x.startswith(("avg_", "min_", "max_")):
        x = x[4:]+"_"+x[0:3]
    if x.find("avg") > 0:
        x = x.replace("_avg","")+"_avg"
    if x.find("win_byts_") > 0:
        x = x.replace("win_byts_", "")+"_win_byts"
    if x.find("init") > 0:
        x = "init_"+x.replace("_init", "")
    if x.endswith("_fwd"):
        x = "fwd_"+x[0:-4]
    else:
        pass
    return x

## fix for massive inconsistencies in CIC-IDS2017 feature set
def sickCICFeatureName(test):
    test = test.to_series().apply(lambda x: x.strip().replace(" ","_").replace("/", "_").casefold()
                                            .replace("destination","dst").replace("source", "src").replace("total", "tot")
                                           .replace("packet", "pkt").replace("length", "len").replace("count","cnt")
                                           .replace("_of_","_").replace("bytes","byts").replace("backward", "bwd")
                                           .replace("tot_len", "totlen").replace("average", "avg").replace("variance", "var")
                                           .replace("segment", "seg").replace("bulk", "b").replace("forward", "fwd")
                                            .replace("_id", "_ID").replace("lab", "Lab").replace("b_rate", "blk_rate")
                                           .replace("data_pkt", "data_pkts").replace("cwe_flag_cnt", "cwe_flag_count"))
    test = [tempoFunc(x) for x in test]
    return test

    
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


# builds a table from tests done previously on same feature set and Attacks/Scan Only/Scanning config.
def buildComparisonTable(scanOnly, scan):
    filepath = "./dissertation/"
    files = [s for s in os.listdir(filepath) if 'fscore_' in s]              # only fscore_ files
    name = "CIC_"
    if scanOnly:
        files = [s for s in files if scatag in s]                            # that have SCAN_
        name = myFunc.scatag + name
    elif not scan:
        files = [s for s in files if atktag in s]                            # OR that have ATK_
        name = myFunc.atktag + name
    else:
        files = [s for s in files if (atktag not in s and scatag not in s)]  # OR that have neither

    table = pd.DataFrame()
    for file in files:#[0:maxNumFiles]:
        temp = pd.read_csv(filepath+file, sep=',')
        table = table.append(temp)
    table.fillna("-", inplace=True)
    
    if not table.empty:
        saveTable( table, '{0}fCross'.format(name),
                  'F1 score of each data set\'s best model on other data sets \[ {0}\]'.format(name.replace("_"," ")),
                  'f1_cross_{0}'.format(name.casefold().replace("_","")) ) 

        
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#----------------#
# LOADING MODELS #
#----------------#

## Get available models to test on a target
def getModelFiles(modelType):
    files = [s for s in os.listdir("./models/") if ( (".joblib" in s) and (modelType in s) )]
    if scatag not in modelType:
        files = [s for s in files if (scatag not in s)]
    if atktag not in modelType:
        files = [s for s in files if (atktag not in s)]
    if "_us" not in modelType:
        files = [s for s in files if ("_us_" not in s)]
    if "_os" not in modelType:
        files = [s for s in files if ("_os_" not in s)]
    return files


# Gets models and table of best performance per model
##
# modelType: name of file .joblib as given by getFilename(pNum, dNum)
##
def loadModel(modelType):
    print("loading models from {0}".format(modelType))
    prep = load( './models/{0}_prep.pkl'.format(modelType) )
    files = getModelFiles(modelType)
    table = pd.DataFrame({},columns=["model","avg_score","avg_fit_time"])
    bestScore = 0
    models = {}
    algo = "ErrorNoFiles"
    print("Models fetched: {0}".format(files))
    for file in files:
        testModel = load('./models/'+file)
        indice = testModel.best_index_
        testline = {"model":file.replace(".joblib","").rsplit("_")[-1],
               "avg_score":testModel.best_score_,
               "avg_fit_time":testModel.cv_results_["mean_fit_time"][indice]}
        print("{0}\'s index of best performance: {1}".format(testline['model'], indice))
        if testline["avg_score"] >= bestScore:
            bestScore = float(testline["avg_score"])
            algo = testline['model']
        table = table.append(testline, ignore_index = True)
        models.update({testline['model']:testModel})
    if bestScore == 0:
        algo = "ErrorBestScore"
    print(algo)
    return (models, prep, table, algo)

    
    
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
def getFeatureList(varType):
    featFile = open("./ML-output/features_{0}.txt".format(getDSName(varType), "r"))
    features = featFile.read()
    featFile.close()
    features = features.split(", ")

    zeroVar = zeroVarRead(varType)
    return [x for x in features if x not in zeroVar]

def loadAndSet(DSName, zeroVarType):
    
    finalfilepath = "./dataset/final/{0}.csv".format( DSName.replace("ATK_","").replace("SCAN_","") )
    
    features = getFeatureList(zeroVarType)
    
    if os.path.isfile(finalfilepath):
        print( "Loading data set from existing file: {0}.csv".format( DSName ) )
        if ("BoT" in DSName) or ("ToN" in DSName):
            data = pd.read_csv(finalfilepath, sep=',', usecols=features)[features]
#            data = data.astype(FEATURE_TYPES)
        else:
            data = pd.read_csv(finalfilepath, dtype=FEATURE_TYPES, sep=',', usecols=features)[features]
        y = data.Label    
        scanTypes = ["reconnaissance", "portscan", "scanning"]
        # Exclude other attacks from data
        if scatag in DSName and "AB-TRAP" not in DSName:
            targetText = scanTypes.copy()
            targetText.append("benign")
            temp = data["Label"].apply(lambda x: True if x.casefold() in targetText else False)
            data = data[temp]
            y = y[temp]
        # Define identification scheme
        targetText = scanTypes.copy()
        targetValue = 1
        #targetToML = (0, 1)
        #index = 1
        if atktag in DSName or "AB-TRAP" in DSName:
            targetText = ["benign"]
            targetValue = 0
        y = y.apply(lambda x: targetValue if x.casefold() in targetText else (1-targetValue))
        y = y.astype('int32')
        
        data.drop('Label', axis=1, inplace=True)
        return data, y
    print("Error: file not found")
    return {}
        
#-----------------#
# LOADING DATASET #
#-----------------#
# TO_DO: None

##
# Returns Loaded or builds required data set, fixing labels and removeing bad lines
#
# INPUT: pNum        [pcap source number: 0 to 4],
#        maxNumFiles [maximum number of files to load, int],
#        dNum        [feature set number: 0 or 1],
#        filepath    [file's directory path, string],
#        BUILD       [wether to build regardless, True/False]
#
# OUTPUT: data       [full data set]
##

def loadDataset(pNum, maxNumFiles, dNum, filepath="./dataset/", BUILD=False):
    finalfilepath = "./dataset/final/{0}.csv".format( getDSName(pNum, dNum) )
    if os.path.isfile(finalfilepath) and not BUILD:
        print( "Loading data set from existing file: {0}.csv".format(getDSName(pNum, dNum)) )
        if pNum in [3, 4]:
            data = pd.read_csv(finalfilepath, sep=',')
            data = data.astype(FEATURE_TYPES)
        else:
            data = pd.read_csv(finalfilepath, dtype=FEATURE_TYPES, sep=',') 
    else:
        if BUILD:
            MSG = "BUILD var set"
        else:
            MSG = "Not Found"
        print( "{1}: building {0}.csv".format(getDSName(pNum, dNum), MSG) )
        data = buildDataset(pNum, maxNumFiles, dNum, filepath)
    return data


# Needs global variables: datasetTypeNum and filepath
#
## loads data from csv files and format output depending on feature set choice and zero variance variables
###
## pcapTypeNum: pcap option number to find proper dataset
## maxNumFiles: maximum number of files to load, in case of fragmented dataset [update in future to maximum total loaded size]
## datasetTypeNum: Feature set option number to find proper dataset
## filepath: Path to dataset repository from step 2
###

# pcapType = {0:"output", 1:"NB15_", 2:"WorkingHours", 3:"ToN-IoT", 4:"BoT-IoT"}

def buildDataset(pcapTypeNum, maxNumFiles, datasetTypeNum, filepath):
    DSName = getDSName(pcapTypeNum, datasetTypeNum) #FINAL DATA SETS CONATIN ALL ATTACK TYPES
    #full_data = pd.DataFrame({}, columns=[])
    temp = []

    # Load files of pcapType and datasetType no more than maxNumFiles
    files = [s for s in os.listdir(filepath) if (datasetType[datasetTypeNum] in s and pcapType[pcapTypeNum] in s)]
    maxNumFiles = min(maxNumFiles, len(files))
    if pcapTypeNum < 2 or pcapTypeNum == 5:
        for file in files[0:maxNumFiles]:
            temp.append(pd.read_csv(filepath+file, dtype=FEATURE_TYPES, sep=','))
    elif pcapTypeNum == 2:
        for file in files[0:maxNumFiles]:
            temp.append(pd.read_csv(filepath+file,dtype={'Flow ID':'string', ' Source IP':'string', ' Destination IP':'string',
                                                         ' Timestamp':'string', ' Label':'string'}, encoding='utf-8', sep=','))
    else:
        for file in files[0:maxNumFiles]:
            temp.append(pd.read_csv(filepath+file, sep=','))
 
    # If AB-TRAP LAN or internet
    if pcapTypeNum == 0:
        # Attack dataset
        temp.append(pd.read_csv(filepath+"attack"+datasetType[datasetTypeNum], dtype=FEATURE_TYPES, sep=','))
    if pcapTypeNum == 5:
        # Attack dataset
        files = [s for s in os.listdir(filepath) if (datasetType[datasetTypeNum] in s and pcapType[0] in s)]
        #maxNumFiles = min(maxNumFiles, len(files)) SORT LATER
        for file in files[0:96]:
            temp.append(pd.read_csv(filepath+file, dtype=FEATURE_TYPES, sep=','))
        #temp = pd.read_csv(filepath+file, sep=',') 
        #full_data = pd.concat([full_data,temp], ignore_index=True)
    # Create AB-TRAP based dataset with all packets (bonafide and attack)
    full_data = pd.concat(temp, ignore_index=True)
        
        #full_data = full_data.astype({'Label':'str'})
        #full_data.loc[full_data['Label']=='benign','Label']='BENIGN'

    if pcapTypeNum == 2:
        full_data.columns = sickCICFeatureName(full_data.columns)
        #full_data.drop(full_data[full_data.isna().any(axis=1)].index, inplace=True) #[['src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol']]
        full_data.drop(['fwd_header_len.1'], axis = 1, inplace = True)
        #full_data.drop(full_data[(full_data == np.inf).any(axis=1)].index, inplace=True)
        #full_data = full_data.astype(FEATURE_TYPES)
        #temp.fillna({temp.columns[temp.isna().any()][0]:0}, inplace=True)

    # fix for ToN-IoT and BoT-IoT name divergence and rogue white spaces [CIC feature set]
    if datasetTypeNum == 1:
        full_data.columns = standardCICFeatureName(full_data.columns)
    
    # ToN and BoT have label and attack as in UNSW-NB15 style data set.
    if pcapTypeNum in [3, 4]:
        full_data.drop(['Label'], axis = 1, inplace = True)
        full_data.rename(columns={'attack': 'Label'}, inplace=True)
        
    #----------------#
    # Drop bad flows #
    #----------------#
    
    flowCount = full_data.shape[0]
    print("Flow count: {0}".format(flowCount))
    
    # Drop full NaN lines
    full_data.drop(full_data[full_data.isna().all(axis=1)].index, axis = 0, inplace = True)
    log(DSName, "Removed {0} lines of full NaN values".format(flowCount-full_data.shape[0]))
    flowCount = full_data.shape[0]
    print("Flow count: {0}".format(flowCount))
    
    # Drop ID NaN lines
    full_data.drop(full_data[full_data[ALL_ID].isna().any(axis=1)].index, axis = 0, inplace = True)
    log(DSName,"Removed {0} lines of NaN ID values".format(flowCount-full_data.shape[0]))
    flowCount = full_data.shape[0]
    print("Flow count: {0}".format(flowCount))
    
    # Drop infinity valued feature lines
    full_data.drop(full_data[(full_data == np.inf).any(axis=1)].index, axis = 0, inplace = True)
    log(DSName,"Removed {0} lines with infinity valued features".format(flowCount-full_data.shape[0]))
    flowCount = full_data.shape[0]
    print("Flow count: {0}".format(flowCount))
    
    # Drop duplicated lines
    full_data.drop(full_data[full_data.duplicated()].index, axis = 0, inplace = True)
    log(DSName,"Removed {0} duplicated lines".format(flowCount-full_data.shape[0]))
    flowCount = full_data.shape[0]
    print("Flow count: {0}".format(flowCount))
    
    full_data.fillna(0, inplace=True)
    
    full_data["dst_port"] = full_data["dst_port"].apply(float)
    full_data = full_data.astype({"dst_port":"int32"})
    
    full_data = full_data.astype(FEATURE_TYPES)
    # Print number of flows and attack/bonafide distribution
    if datasetTypeNum == 0:
        # if NB15 feature set: data['Label'] == 0
        columnName = 'Label'
        columnValue = 0
    if datasetTypeNum == 1:
        # if CIC feature set: data['Label'] == 'benign'
        columnName = 'Label'
        columnValue = 'benign'
    #print( "DEBUG: ", full_data[columnName].unique() )#apply(lambda x: x.casefold()) )
    examples_bonafide = full_data[full_data[columnName].apply(lambda x: True if x.casefold() == columnValue else False)].shape[0] #examples_bonafide = full_data[full_data[columnName] == columnValue].shape[0]
    total = full_data.shape[0]
    log(DSName,'Total examples of {0} with {1} attacks and {2} bonafide flows'.format(total, total - examples_bonafide, examples_bonafide))

    # check features with zero variance (not useful for learning) and general ID features
    zeroVar = full_data.select_dtypes(exclude='string').columns[(full_data.var() == 0).values]
    zeroVar = np.concatenate((zeroVar.values.T, ID_FEATURES))
    zeroVarWrite(zeroVar,pcapTypeNum)         
    
    print("saving finalized dataset")
    full_data.to_csv("./dataset/final/{0}.csv".format( DSName ), index=None, header=True)
       
    return full_data


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#----------------#
# PREPARE FOR ML #
#----------------#
# TO_DO: Add misssing dNum variable for other feature sets

##
# Returns X, y for training models
#
# INPUT: full_data   [data set as source: DataFrame],
#        pNum        [pcap source number: 0 to 4],
#        scanOnly    [remove other attacks: True/False],
#        scan        [only detect scan attack: True/False],
#        zeroVarType [zero variance list: 0 to 4]
#
# OUTPUT: X          [feature set values per flow],
#         y          [0 or 1 per flow]
##
def setTarget(full_data, pNum, scanOnly, scan, zeroVarType):
    #---------------#
    # DEFINE TARGET #
    #---------------#
    zeroVar = zeroVarRead(zeroVarType)
    full_data.drop(columns=zeroVar, axis=1, inplace=True)
    
    X = full_data.drop(columns = ["Label"])
    y = full_data.Label
    scanTypes = ["reconnaissance", "portscan", "scanning"]
    # Exclude other attacks from data
    if scanOnly and pNum:
        targetText = scanTypes.copy()
        targetText.append("benign")
        temp = full_data["Label"].apply(lambda x: True if x.casefold() in targetText else False)
        X = X[temp]
        y = y[temp]
        log(getDSName(pNum, 1, scanOnly, scan),"setTarget: Removed {0} flows from other attack types".format(full_data.shape[0] - X.shape[0]))
    # Define identification scheme
    targetText = ["benign"]
    targetToML = (0, 1)
    index = 0
    if not scanOnly and scan and pNum:
        targetText = scanTypes.copy()
        index = 1
    y = y.apply(lambda x: targetToML[index] if x.casefold() in targetText else targetToML[index-1])
    y = y.astype('int32')
    
    return X, y