#----------------------------------------------------------------------------------------
#
#                                      NB15_dataset.py
#
#
# Input: Argus(argus.csv), Zeek(conn.log, http.log, ftp.log), Labels(attack_label.csv)
# Ouput: (${PCAP}_NB15.csv)
#
# Discription:
# Extract 49 features as in (Moustafa et al 2015) from ${PCAP}.pcap file
#-----------------------------------------------------------------------------------------


# DATASET INFO AND PCAP FILE SOURCES

## UNSW-NB15
### source: https://research.unsw.edu.au/projects/unsw-nb15-dataset

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import socket
from datetime import datetime


#DS = pd.DataFrame(np.empty((0, 49)))     # 49 empty columns
DS = []
 
    

## convert to int
def portsAsInt(x):
    if isinstance(x,str):     #if is string
        if x.isnumeric():        #and if contains only decimals
            return int (x)
        else:
            try:
                return int(float(x))
            except ValueError:
                return int(x,16) #if contains hex number
    return 0

    
## get argus.csv, conn.log, http.log and ftp.log, merge/format to NB15 dataset and calculate 12 additional featues
def toNB(labelType, pcapName, filepath):   
    
    global DS
    
    # load the CSVs from a specific pcap file
    HAS_CONN = os.path.isfile(filepath+"conn.log")
    HAS_HTTP = os.path.isfile(filepath+"http.log")
    HAS_FTP = os.path.isfile(filepath+"ftp.log")

    print("loading argus.csv..")
    DS = pd.read_csv(filepath+"argus.csv")                                             # dataset Argus
    print("argus.csv", DS.shape)

    if(HAS_CONN):
        print("loading conn.log..")
        zconn = pd.read_csv(filepath+"conn.log", sep='\t', skiprows = [0, 1, 2, 3, 4, 5, 7]) # dataset Zeek Conn
        zconn.columns = np.concatenate([zconn.columns[1:], ['drop']])                 # mark extra column for drop
        zconn.drop('drop', axis = 1, inplace = True)                                  # drop marked column
        print("conn.log", zconn.shape)
    else:
        print("no argus.csv")

    if(HAS_HTTP):
        print("loading http.log..")
        zhttp = pd.read_csv(filepath+"http.log", sep='\t', skiprows = [0, 1, 2, 3, 4, 5, 7]) # dataset Zeek http
        zhttp.columns = np.concatenate([zhttp.columns[1:], ['drop']])                 # mark extra column for drop
        zhttp.drop('drop', axis = 1, inplace = True)                                  # drop marked column
        print("http.log", zhttp.shape)
    else:
        print("no http.log")
    # trans_depth and response_body_len

    if(HAS_FTP):
        print("loading ftp.log..")
        zftp = pd.read_csv(filepath+"ftp.log", sep='\t', skiprows = [0, 1, 2, 3, 4, 5, 7])   # dataset Zeek ftp
        zftp.columns = np.concatenate([zftp.columns[1:], ['drop']])                   # mark extra column for drop
        zftp.drop('drop', axis = 1, inplace = True)                                   # drop marked column
        print("ftp.log", zftp.shape)
    else:
        print("no ftp.log")

    DS.drop(DS[DS['Proto']=='man'].index, axis=0, inplace=True)
    DS.reset_index(drop=True, inplace=True)
        
        
    #-------#
    # ARGUS #
    #-------#
    
    #Format argus.csv: data fix port and time parsing, uses portsAsInt(x)
    DS = DS.astype({'SrcAddr':'string', 'Sport':'string', 'DstAddr':'string', 'Dport':'string', 'Proto':'string', 'State':'string'})
    DS['Dport'] = DS['Dport'].apply(lambda x: portsAsInt(x))
    DS['Sport'] = DS['Sport'].apply(lambda x: portsAsInt(x))
    DS[['Sport','Dport']].fillna(0, inplace=True)

    if (DS['Dport'].notna().all() and DS['Sport'].notna().all()):
        if (DS['Dport'].apply(lambda x: isinstance(x,int)).all() and DS['Sport'].apply(lambda x: isinstance(x,int)).all()):
            print("all ports are properly parsed")
        else:
            print("not all port properly parsed")
    else:
        print("some ports are NA")

    DS = DS.astype({'SrcAddr':'string', 'Sport':'int32', 'DstAddr':'string', 'Dport':'int32', 'Proto':'int32', 'State':'string'})
    if isinstance(DS['StartTime'][0],str):
        DS['StartTime'] = DS['StartTime'].apply(lambda x: float(x))
        DS['LastTime'] = DS['LastTime'].apply(lambda x: float(x))
   


    #----------#
    # CONN.LOG #
    #----------#

   
    if HAS_CONN:
        #Format conn.log data
        if zconn.columns.isin(['id.orig_h','id.orig_p','id.resp_h','id.resp_p']).any():
            badIndex = zconn[['id.orig_p','id.resp_p']].isna().all(axis=1)
            badIndex = badIndex[badIndex].index
            zconn.drop(badIndex, axis=0, inplace=True)
            zconn.reset_index(drop=True, inplace=True)
            zconn = zconn.astype({'id.orig_h':'string', 'id.orig_p':'int32', 'id.resp_h':'string', 'id.resp_p':'int32', 'proto':'string','service':'string'})

        zconn.columns = ['StartTime', 'uid', 'SrcAddr', 'Sport', 'DstAddr','Dport','Proto', 'service', 'duration', 'orig_bytes', 'resp_bytes','conn_state',
                         'local_orig', 'local_resp', 'missed_bytes', 'history','orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes','tunnel_parents']
        # may be removed -- adapt argus protocol display for compatibility instead
        protList = {'tcp':'6', 'udp': '17', 'ipv4': '4', 'icmp': '1', 'igmp': '2' }
        zconn.['Proto'] = zconn.['Proto'].apply(lambda x: protoList[x])
#        test = zconn['Proto']
#        for loc in test.index:
#            if not(str(test.iloc[loc]).isnumeric()):
#                if test.iloc[loc] == "tcp":
#                    zconn['Proto'].iloc[loc] = '6'
#                if test.iloc[loc] == "udp":
#                    zconn['Proto'].iloc[loc] = '17'
#                if test.iloc[loc] == "ipv4":
#                    zconn['Proto'].iloc[loc] = '4'
#                if test.iloc[loc] == "icmp":
#                    zconn['Proto'].iloc[loc] = '1'
#                if test.iloc[loc] == "igmp":
#                    zconn['Proto'].iloc[loc] = '2'
        zconn = zconn.astype({'Proto':'int32'})
        zconn['StartTime'] = zconn['StartTime'].apply(lambda x: float(x))
#        print("Unique protocol list :", zconn['Proto'].unique())
        
        # Merging data from conn.log
        zconn.drop(['uid', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp', 'missed_bytes', 'history','orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes','tunnel_parents'], axis = 1, inplace = True)
        DS = DS.merge(zconn, how='left',
                      left_on=['SrcAddr', 'Sport','DstAddr','Dport','Proto','StartTime'],
                      right_on=['SrcAddr', 'Sport', 'DstAddr', 'Dport','Proto','StartTime'])
        DS.fillna(value={'service': '-','duration': 0,'conn_state': '-'}, inplace=True)
    else:
        print("No conn.log")
        DS[['service','duration','conn_state']] = ['-',0,'-']
 
    

    #----------#
    # HTTP.LOG #
    #----------#
    
    
    if HAS_HTTP:
        # Formating http.log
        if zhttp.columns.isin(['ts','id.orig_h','id.orig_p','id.resp_h','id.resp_p']).any():
            zhttp.columns = ['StartTime', 'uid', 'SrcAddr', 'Sport', 'DstAddr','Dport','trans_depth', 'method', 'host', 'uri', 'referrer','version',
                             'user_agent','origin', 'request_body_len', 'response_body_len','status_code', 'status_msg', 'info_code', 'info_msg','tags',
                             'username', 'password', 'proxied', 'orig_fuids', 'orig_filenames','orig_mime_types', 'resp_fuids', 'resp_filenames', 
                             'resp_mime_types']
        badIndex = zhttp[['Sport','Dport']].isna().all(axis=1)
        badIndex = badIndex[badIndex].index
        zhttp.drop(badIndex, axis=0, inplace=True)
        zhttp.reset_index(drop=True, inplace=True)
        zhttp['service'] = 'http'
        zhttp['Proto'] = 6
        zhttp = zhttp.astype({'StartTime':'float','SrcAddr':'string', 'Sport':'int32','DstAddr':'string','Dport':'int32','Proto':'int32','service':'string',
                              'trans_depth':'int32','response_body_len':'int32','method':'string'})
        # Merging data from http.log (port 80)
        zhttp.drop(['host', 'uri', 'referrer','version', 'user_agent','origin', 'request_body_len', 'status_code', 'status_msg', 'info_code',
                    'info_msg','tags', 'username', 'password', 'proxied', 'orig_fuids', 'orig_filenames','orig_mime_types', 'resp_fuids',
                    'resp_filenames', 'resp_mime_types'], axis = 1, inplace = True)
        DS = DS.merge(zhttp, how='left',
                       left_on=['SrcAddr', 'Sport', 'DstAddr', 'Dport','Proto','service'],
                       right_on=['SrcAddr', 'Sport', 'DstAddr', 'Dport','Proto','service'])
       
        DS.fillna(value={'trans_depth': 0,'response_body_len': 0,'method': '-'}, inplace=True)
    else:
        print("No http.log")
        DS[['trans_depth','response_body_len','method']] = [0,0,'-'] 
    
    
   
    #---------#
    # FTP.LOG #
    #---------#
    
    if HAS_FTP: 
        # Formating ftp.log data
        if zftp.columns.isin(['id.orig_h','id.orig_p','id.resp_h','id.resp_p']).any():
            zftp.columns = ['StartTime', 'uid', 'SrcAddr', 'Sport', 'DstAddr','Dport','user','password','command','arg','mime_type','file_size','reply_code',
                            'reply_msg','data_channel.passive','data_channel.orig_h','data_channel.resp_h','data_channel.resp_p','fuid']
        badIndex = zftp[['Sport','Dport']].isna().all(axis=1)
        badIndex = badIndex[badIndex].index
        zftp.drop(badIndex, axis=0, inplace=True)
        zftp.reset_index(drop=True, inplace=True)
        zftp['service'] = 'ftp'
        zftp['Proto'] = 6
        zftp = zftp.astype({'StartTime':'float','SrcAddr':'string', 'Sport':'int32', 'DstAddr':'string','Dport':'int32',
                            'Proto':'int32','service':'string','user':'string','password':'string', 'command':'string'})
        
        # Merging data from ftp.log (port 21)
        zftp.drop(['StartTime', 'uid','arg','mime_type','file_size','reply_code','reply_msg','data_channel.passive',
                   'data_channel.orig_h','data_channel.resp_h','data_channel.resp_p','fuid'], axis = 1, inplace = True)
        DS = DS.merge(zftp, how='left',
                        left_on=['SrcAddr', 'Sport', 'DstAddr', 'Dport','Proto','service'],
                        right_on=['SrcAddr', 'Sport', 'DstAddr', 'Dport','Proto','service'])
        print("Flows in DS: ", DS.shape[0], "\nFlows in ftp.log: ", zftp.shape[0],)
        print("Non repeated in zftp", zftp[zftp.duplicated(subset=['SrcAddr', 'Sport', 'DstAddr', 'Dport','Proto','service'], keep='first')].shape[0])
        DS.fillna(value={'user': '-','password': '-','command': '-'}, inplace=True)
        DS[DS['service']=='ftp'].head(5)
    else:
        print("No ftp.log")
        DS[['user','password','command']] = ['-','-','-']    
    
    

    #-------------------------------#
    # Fitting into UNSW-NB15 format #
    #-------------------------------#
    DS = DS[['SrcAddr', 'Sport', 'DstAddr', 'Dport', 'Proto', 'State', 'Dur','SrcBytes', 'DstBytes', 'sTtl', 'dTtl',
               'SrcLoss', 'DstLoss','service', 'SrcLoad', 'DstLoad', 'SrcPkts', 'DstPkts', 'SrcWin', 'DstWin', 'SrcTCPBase',
               'DstTCPBase', 'sMeanPktSz', 'dMeanPktSz', 'trans_depth','response_body_len', 'SrcJitter', 'DstJitter','StartTime',
               'LastTime', 'SIntPkt', 'DIntPkt', 'TcpRtt', 'SynAck', 'AckDat', 'Trans', 'Min',
               'Max', 'Sum', 'duration', 'conn_state', 'method', 'user', 'password', 'command']]
    DS.columns = ['srcip', 'sport', 'dstip', 'dport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
                   'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb',
                   'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime',
                   'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat', 'Trans', 'Min',
                   'Max', 'Sum', 'duration', 'conn_state', 'method', 'user', 'password', 'command']

    #--------------------------#
    # General Purpose Features #
    #--------------------------#

    print("calculating General Purpose Features..")
    #'is_sm_ips_ports'
    DS.loc[DS['srcip'] == DS['dstip'], 'is_sm_ips_ports'] = 1
    #DS['is_sm_ips_ports'].replace(to_replace={True: 1, False: 0}, inplace=True)
    DS.fillna(value={'is_sm_ips_ports': 0}, inplace=True)

    # 'ct_state_ttl'
    test = DS.groupby(['state','sttl','dttl'], as_index=False).size()
    for line in test.index:
        DS.loc[ (DS['state'] == test.iloc[line,0]) & (DS['sttl'] == test.iloc[line,1]) &
               (DS['dttl'] == test.iloc[line,2]), 'ct_state_ttl'] = test.iloc[line,3]
    DS.fillna(value={'ct_state_ttl': 0}, inplace=True)
    DS['ct_state_ttl'] = DS['ct_state_ttl'].apply(int)

    # 'ct_flw_http_mthd' 
    test = DS.groupby(['method'], as_index=False).size()
    test.loc[ test['method'] == '-', 'size'] = 0
    for line in test.index:
        DS.loc[ DS['method'] == test.iloc[line,0] , 'ct_flw_http_mthd'] = test.iloc[line,1]

    DS.fillna(value={'ct_flw_http_mthd': 0}, inplace=True)
    DS['ct_flw_http_mthd'] = DS['ct_flw_http_mthd'].apply(int)

    # is_ftp_login
    DS['is_ftp_login'] = '-'
    DS.loc[ (DS['user'] == '-') | (DS['user'] == '<unknown>') | (DS['user'] == 'anonymous') | (DS['password'] == '-'), 'is_ftp_login'] = 0
    DS.loc[ (DS['is_ftp_login'] != 0) & (DS['service'] == 'ftp'), 'is_ftp_login'] = 1
    
    # ct_ftp_cmd
    test = DS[DS['service']=='ftp'].groupby(['srcip','dstip','sport','dport','command'], as_index=False).size()
    test.drop(index=test[test['command']=='-'].index, inplace=True)
    #test = test.groupby(['srcip','dstip','sport','dport'], as_index=False).size()
    test['service'] = 'ftp'
    test.rename(columns={"size":"ct_ftp_cmd"}, inplace=True)
    DS = DS.merge(test, how='left', left_on=['srcip','dstip','sport','dport','service'],
                  right_on=['srcip','dstip','sport','dport','service'])
    DS.fillna(value={'ct_ftp_cmd': 0}, inplace=True)
    DS['ct_ftp_cmd'] = DS['ct_ftp_cmd'].apply(int)

    #---------------------#
    # Connection Features #
    #---------------------#
    
    print("calculating Connection Features..")
    DS.sort_values('ltime', inplace=True, kind='mergesort', ignore_index=True)

    # trying to make it faster
    simpleDS = DS[['srcip','dstip','sport','dport','service']]#.values
    featuresNames = ['ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
    # ct_srv_src
    simpleDS[featuresNames[0]] = (simpleDS['srcip']+'-'+simpleDS['service'].apply(str)).values
    # ct_srv_dst
    simpleDS[featuresNames[1]] = (simpleDS['dstip']+'-'+simpleDS['service'].apply(str)).values
    # ct_dst_ltm
    simpleDS[featuresNames[2]] = simpleDS['dstip'].values
    # ct_src_ltm
    simpleDS[featuresNames[3]] = simpleDS['srcip'].values
    # ct_src_dport_ltm
    simpleDS[featuresNames[4]] = (simpleDS['srcip']+'-'+simpleDS['dport'].apply(str)).values
    # ct_dst_sport_ltm
    simpleDS[featuresNames[5]] = (simpleDS['dstip']+'-'+simpleDS['sport'].apply(str)).values
    # ct_dst_src_ltm
    simpleDS[featuresNames[6]] = (simpleDS['srcip']+'-'+simpleDS['dstip']).values
    
    simpleDS = simpleDS[featuresNames]
    countDS = simpleDS
    countDS.iloc[0] = np.zeros((1,len(featuresNames)),dtype='int')
    
    for indice in range(1,len(DS.index)):
        countDS.iloc[indice] = simpleDS.iloc[indice-min(indice,100):indice].isin(simpleDS.iloc[indice].values).sum().values
    for i in featuresNames:
        DS[i] = countDS[i]
    
   
    #--------#
    # LABELS #
    #--------#
    
    # loading Labels
    print("labeling...")
    DS.drop(['Trans', 'Min', 'Max', 'Sum', 'duration', 'conn_state', 'method', 'user', 'password', 'command'], axis = 1, inplace = True)
    
    # Bonafide.pcap - all benign
    if labelType == 1:
        DS['attack_cat'] = ''
        
    # attack.pcap - attack_label.csv (ip, label) #label means 'attack category', not 'attack/benign'
    if labelType == 2:
        labels = pd.read_csv(filepath + "attack_label.csv")

        # insert attack category and label
        labels.rename(columns={'ip':'srcip', 'label':'attack_cat'}, inplace=True)
        labels['attack_cat'] = labels['attack_cat'].str.strip()
        DS = DS.merge(labels[['srcip','attack_cat']],
                      how='left',
                      left_on=['srcip'],
                      right_on=['srcip'])
    
    # nb15.pcap - NUSW-NB15_GT.csv (Source IP, Destination IP, Source Port, Destination Port, Protocol, Attack category)
    if labelType == 3:
        labels = pd.read_csv(filepath + "NUSW-NB15_GT.csv")

        # insert attack category and label
        labels.rename(columns={'Source IP':'srcip', 'Destination IP':'dstip', 'Source Port':'sport',
                                     'Destination Port':'dport', 'Protocol':'proto', 'Attack category':'attack_cat'}, inplace=True)
    
    # cic.pcap (Source IP, Source Port, Destination IP, Destination Port, Protocol, Label)
    if labelType == 4:
        # single label file need to be with the same name as the folder or optional name
        labels = pd.read_csv(filepath + pcapName+".csv")

        # insert attack category and label
        labels.rename(columns={'Source IP':'srcip', 'Destination IP':'dstip', 'Source Port':'sport',
                                     'Destination Port':'dport', 'Protocol':'proto', 'Label':'attack_cat'}, inplace=True)
    if labelType in [3, 4]:
        labels['attack_cat'] = labels['attack_cat'].str.strip()
        DS = DS.merge(labels[['srcip','dstip','sport','dport','proto', 'attack_cat']],
                      how='left',
                      left_on=['srcip','dstip','sport','dport', 'proto'],
                      right_on=['srcip','dstip','sport','dport', 'proto'])
        
        
    DS.fillna(value={'attack_cat': ''}, inplace=True)
    DS['Label'] = 1
    DS.loc[DS['attack_cat'] == '','Label'] = 0
    
    #--------------#
    # SAVE DATASET #
    #--------------#
    
    print("saving..")
    DS.fillna(value={'sttl': 0, 'dttl': 0, 'swin': 0, 'dwin': 0, 'stcpb': 0, 'dtcpb': 0, 'sjit': 0, 'djit': 0,'dintpkt': 0}, inplace=True)
    if DS.columns.isin(['Trans', 'Min', 'Max', 'Sum', 'duration', 'conn_state', 'method', 'user', 'password', 'command']).any():
        DS.drop(['Trans', 'Min', 'Max', 'Sum', 'duration', 'conn_state', 'method', 'user', 'password', 'command'], axis = 1, inplace = True)
    DS.to_csv("./dataset/" + pcapName + '_NB15.csv', index=None, header=True)

    
if __name__ == "__main__":
    filepath = "./csv/"
    
    # help
    if len(sys.argv) < 3:
        print("Usage: " + sys.argv[0] + " <TYPE_LABEL> <PATH_TO_CSV> [OUTPUT_NAME]")
        print("Types of labels are \n1 - Bonafide\n2 - Attack\n3 - NB15\n4 - CIC")
        sys.exit()
        
    if len(sys.argv)>2:
        labelType = sys.argv[1]
        
        # check for invalid types
        if (labelType > 4) or (labelType < 1):
            print("Types of labels are \n1 - Bonafide\n2 - Attack\n3 - NB15\n4 - CIC")
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
    if not os.path.isfile(filepath + "argus.csv"):
        print("missing file: ", filepath + "argus.csv")
        sys.exit()
    if not os.path.isfile(filepath + "attack_label.csv"):
        print("missing file: ", filepath + "attack_label.csv")
        sys.exit()
        
    toNB(labelType, pcapName, filepath)