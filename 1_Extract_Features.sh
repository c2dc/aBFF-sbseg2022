#!/bin/bash

fileList=($(ls ${1:-"./"}*.pcap))
fileList=${fileList[@]}

PATH=$PATH':/usr/local/zeek/bin:~/.local/bin'

pCheck=`which python`
if [ -z "$pCheck" ]
then
  echo "ERROR: This script requires python."
  exit 255
fi

pCheck=`which argus`
if [ -z "$pCheck" ]
then
  echo "ERROR: This script requires Argus."
  exit 255
fi

pCheck=`which zeek`
if [ -z "$pCheck" ]
then
  echo "ERROR: This script requires Zeek."
  exit 255
fi

pCheck=`which cicflowmeter`
if [ -z "$pCheck" ]
then
  echo "ERROR: This script requires CICFlowMeter."
#  exit 255
fi

if  [ ! -e ./csv ]
then
    mkdir ./csv
fi

echo Reading files: ${fileList}
echo 

for file in ${fileList}
do
	name=${file:0:-5}
	name=${name##*/}
	echo "File: $name"
	echo

	if  [ ! -e ./csv/$name ]
	then
		mkdir ./csv/$name
	fi

	echo "reading PCAP 2 Argus"
	argus -J -r ./$file -w ./csv/$name/$name.argus

	echo "reading Argus 2 CSV"
	ra -nn -u -r ./csv/$name/$name.argus -c ',' -s saddr sport daddr dport proto state dur sbytes dbytes sttl dttl sloss dloss service sload dload spkts dpkts swin dwin stcpb dtcpb smeansz dmeansz sjit djit stime ltime sintpkt dintpkt tcprtt synack ackdat trans min max sum -M dsrs=+time,+flow,+metric,+agr,+jitter > ./csv/$name/argus.csv

	cd ./csv/$name
	echo "reading Zeek"
	zeek -C -r ../../$file
	cd ../../

	#echo "reading CICFlowMeter"
	#cicflowmeter -f ./$file -c ./csv/$name/cic.csv
	echo "File: $name complete"
	echo
done

echo "Done!"
