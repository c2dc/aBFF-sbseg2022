#----------------------------------------------------------------------------------------
#
#                                      featuresCIC.py
#
#
# Input: CICFlowMeter(cic.csv), Labels()
# Ouput: (${PCAP}_CIC.csv)
#
# Discription:
# Extract 84 features as in (Sharafaldin et al 2018) from ${PCAP}.pcap file
#-----------------------------------------------------------------------------------------


# DATASET INFO AND PCAP FILE SOURCES

## CIC-IDS
### source: https://www.unb.ca/cic/datasets/ids-2017.html 

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import myFunc

## set port number as int
def portsAsInt(x):
    try:   #if is string
        if x.isnumeric():
            return int(x)    #if contains only decimals
        else:
            return int(x,16) #if contains hex number
    except:
        return np.NaN

## set protocol number as int by name
def protoNum(proto):
    trad = {"tcp":6, "udp":17, "ipv4":4, "icmp":1, "igmp":2, "ggp": 3, "ip": 0, "egp": 8, "pup": 12, "hmp": 20,
            "xns-idp": 22, "rdp": 27, "ipv6": 41, "ipv6-frag": 44, "ipv6-route": 43, "rvd": 66, "ipv6-opts": 60, "l2tp": 1701}
    if proto.isnumeric():
        return int(proto)
    if proto not in trad.keys():
        return np.NaN
    return trad[proto]


## returns flow ID string
def flowID(dataframe):
    return (dataframe['dst_ip'].apply(str) + '-' + dataframe['dst_port'].apply(str) + '-' + 
            dataframe['src_ip'].apply(str) + '-' + dataframe['src_port'].apply(str) + '-' + dataframe['protocol'].apply(str))


 #----------------#
 # Build data set #
 #----------------#

## get cic.csv, merge/format labels to CIC-IDS dataset.
### Adds labels, fix label formatting
### UNB15: (1) has protocol from argus option (NOT USING CURRENTLY), (2) sets port and protocol as int.
def toCIC(labelType, pcapName, filepath, tuple4=False):   
    
    cicfm = pd.read_csv(filepath + "cic.csv", sep=',') # dataset CICFlow Meter
    cicfm['flow_ID'] = flowID(cicfm)

    #--------#
    # LABELS #
    #--------#
    
    # Bonafide.pcap - all benign
    if labelType == 1:
        cicfm['Label'] = 'BENIGN'
        
    # attack.pcap - attack_label.csv (ip, label) #label means 'attack category', not 'attack/benign'
    if labelType == 2:
        labels = pd.read_csv("./labels/attack_labels.csv",header=0, names=['src_ip', 'Label'])

        # insert attack category and label
        # labels had formatting issues, blanks in weird places
        labels['Label'] = labels['Label'].str.strip()
        cicfm = cicfm.merge(labels,
                      how='inner', #'left', changed due to issue with extra ips on cic and argus files. May be from simulation itself
                      on=['src_ip'])
    
        
    # nb15.pcap - NUSW-NB15_GT.csv (Source IP, Destination IP, Source Port, Destination Port, Protocol, Attack category)
    if labelType == 3:
        if tuple4:
            # Use argus file to solve CICFlowMeter's protocol number issue
            argus = pd.read_csv(filepath+"argus.csv", usecols=['SrcAddr','DstAddr','Sport','Dport','Proto'],
                    dtype={'SrcAddr':'string','DstAddr':'string'},
                    converters={'Proto':protoNum, 'Sport':portsAsInt ,'Dport':portsAsInt}, sep=',') # to fix protocol number in cic
        
            argus.drop(argus[argus.isna().any(axis=1)].index, axis = 0, inplace = True)
            argus.rename(columns={'SrcAddr':'src_ip','DstAddr':'dst_ip','Sport':'src_port','Dport':'dst_port','Proto':'protocol'}, 
                         inplace=True)
            argus = argus.astype({'src_port':'int32', 'dst_port':'int32','protocol':'int32'})
        
        # Get right protocol number from argus file
            cicfm.drop(['protocol'], axis = 1, inplace = True)
            cicfm = cicfm.merge(argus,
                      how='left',
                      on=['src_ip', 'dst_ip', 'src_port', 'dst_port'])
            
        cicfm.drop(cicfm[cicfm[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']].isna().any(axis=1)].index, axis = 0, inplace = True)
        cicfm.drop_duplicates(inplace=True)
        
        # Start labeling from here
        labels = pd.read_csv("./labels/NUSW-NB15_GT.csv",
                             usecols=['Source IP','Destination IP','Source Port','Destination Port','Protocol','Attack category'],
                             dtype={'Source IP':'string', 'Destination IP':'string', 'Attack category':'string'},
                             converters={'Protocol':protoNum}, sep=',')
        # set names for compatibility
        labels.rename(columns={'Source IP':'src_ip', 'Destination IP':'dst_ip', 'Source Port':'src_port',
                                     'Destination Port':'dst_port', 'Protocol':'protocol', 'Attack category':'Label'}, inplace=True)
        # drop lines that cannot be used for labeling
        labels.drop(labels[labels.isna().any(axis=1)].index, axis = 0, inplace = True)
        labels = labels.astype({'src_port':'int32', 'dst_port':'int32', 'protocol':'int32'})
        labels['Label'] = labels['Label'].str.strip()
        
        cicfm = cicfm.merge(labels, how='left', on=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'])  
        cicfm.drop_duplicates(inplace=True)
        cicfm = cicfm.astype({'protocol':'int32'})

        pcapName = "NB15_"+pcapName
        
        
    ## CIC.pcap [not implemented yet]
    if labelType == 4:
        pass
    
    
    ## attack.pcap from internet
    if labelType == 5:
        cicfm['Label'] = 'reconnaissance'
    
    # All still not set are Benign
    cicfm.fillna(value={'Label': 'BENIGN'}, inplace=True)
    
    #--------------#
    # SAVE DATASET #
    #--------------#
    
    cicfm.to_csv("./dataset/" + pcapName + '_CIC.csv', index=None, header=True)
    
if __name__ == "__main__":
    #filepath = "./csv/"
    
    # help
    if len(sys.argv) < 3:
        print("Usage: " + sys.argv[0] + " <TYPE_LABEL> <PATH_TO_CSV> [OUTPUT_NAME]")
        print("Types of labels are \n1 - Bonafide\n2 - Attack\n3 - NB15\n4 - CIC")
        sys.exit()
        
    if len(sys.argv)>2:
        labelType = int(sys.argv[1])
        
        # check for invalid types
        if (labelType > 5) or (labelType < 1):
            print("Types of labels are \n1 - Bonafide\n2 - AttackLAN\n3 - NB15\n4 - CIC\n5 - AttackInternet")
            sys.exit()
        filepath = sys.argv[2]
        
        # check for missing '/'
        if filepath[len(filepath)-1] != '/':
            filepath += '/'
        pcapName = os.path.basename(os.path.dirname(filepath))
      
    # optional name setting
    if len(sys.argv)>3:
        pcapName = sys.argv[3]
    
    # check for missing files
    if not os.path.isfile(filepath + "cic.csv"):
        print("missing file: ", filepath + "cic.csv")
        sys.exit()
        
    toCIC(labelType, pcapName, filepath)
    print("Done!")