#----------------------------------------------------------------------------------------
#
#                                      explain.py
#
#
# Input: trainDataset(${PCAP}_CIC.csv)
# Ouput: Models(${PCAP}_CIC_${ML}.joblib)
#
# Discription:
# Explain models with trainDataset
#-----------------------------------------------------------------------------------------

import sys
import seaborn as sns
import datetime
import pandas as pd
import numpy as np

import shap # v0.39.0

import matplotlib.pyplot as plt
import supportFiles.myFunc as myF



def explainModel(trainerDS, testerDS, mName="SVM"):
    
    shap.initjs()
    
    # Set names
    trainerDSName = myF.getDSName(trainerDS,1,True,True)
    testerDSName = myF.getDSName(testerDS,1,True,True)

    # Get model
    modelList, prep, table, algo = myF.loadModel(trainerDSName)
    model = modelList[mName]

    # Get test target
    X_test, y_test = myF.setTarget(myF.loadDataset(testerDS, 96, 1), testerDS, True, True, trainerDS)

    explainer = shap.Explainer(model.best_estimator_)
    shap_test = explainer(X_test)
    print(f"Shap values length: {len(shap_test)}\n")
    print(f"Sample shap value:\n{shap_test[0]}")
    del X_test

    #bar plot
    shap.plots.bar(shap_test, max_display=18, show=False)
    plt.savefig("./dissertation/explain_{0}_{1}_{2}_DS_bar.jpg".format(trainerDSName, mName, testerDSName), bbox_inches="tight")
    plt.close()

    #violin plot
    shap.summary_plot(shap_test, max_display=11, show=False, plot_type='violin') #plot_size="auto"/(11,8)
    plt.savefig("./dissertation/explain_{0}_{1}_{2}_DS_violin.jpg".format(trainerDSName, mName, testerDSName), bbox_inches="tight")
    plt.close()

# IF CALLED FROM TERMINAL

if __name__ == "__main__":

    datasetMSG = "Datasets available are :\n"

    #Datasets available are :
    # {0: 'AB-TRAP', 1: 'NB15', 2: 'CIC-IDS', 3: 'ToN-IoT', 4: 'BoT-IoT', 5:"internet"}

    # help
    if len(sys.argv) < 4:
        print("Usage: " + sys.argv[0] + " <TRAINER_DATA_SET> <TESTER_DATA_SET> <ML_MODEL_NAME>")
        print(datasetMSG, myF.datasetOptions())
        sys.exit()
        
    trainerDS = int(sys.argv[1])
    testerDS = int(sys.argv[2])
    mName = sys.argv[3]
    
    explainModel(trainerDS, testerDS, mName)
    
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX