#----------------------------------------------------------------------------------------
#
#                                      CIC_labels.py
#
#
#
# Discription:
# Format CIC label files for subsequent use
#-----------------------------------------------------------------------------------------

import os
import sys
import pandas as pd
import warnings
import shutil
warnings.filterwarnings('ignore')
    
def main():   
    
    filepath = "./labels/"
    labels = pd.read_csv(filepath + "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", sep=',')
    labels = labels.append( pd.read_csv(filepath + "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", sep=','), ignore_index=True)
    labels = labels.append( pd.read_csv(filepath + "Friday-WorkingHours-Morning.pcap_ISCX.csv", sep=','), ignore_index=True)
    labels.to_csv('./dataset/Friday-WorkingHours_CIC.csv', index=None, header=True)
    
    #labels = pd.read_csv(filepath + "Monday-WorkingHours.pcap_ISCX.csv", sep=',')
    shutil.copy(filepath + "Monday-WorkingHours.pcap_ISCX.csv", "./dataset/Monday-WorkingHours_CIC.csv")
    
    labels = pd.read_csv(filepath + "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", sep=',')
    labels = labels.append( pd.read_csv(filepath + "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", sep=','), ignore_index=True)
    labels.to_csv('./dataset/Thursday-WorkingHours_CIC.csv', index=None, header=True)
    
    #labels = pd.read_csv(filepath + "Tuesday-WorkingHours.pcap_ISCX.csv", sep=',')
    shutil.copy(filepath + "Tuesday-WorkingHours.pcap_ISCX.csv","./dataset/Tuesday-WorkingHours_CIC.csv")
                
    #labels = pd.read_csv(filepath + "Wednesday-workingHours.pcap_ISCX.csv", sep=',')
    shutil.copy(filepath + "Wednesday-WorkingHours.pcap_ISCX.csv","./dataset/Wednesday-WorkingHours_CIC.csv")
    
if __name__ == "__main__":
    main()