# ABOUT FEATURES
## FEATURES EXTRACTION
- NB-15 extracts features from Argus and Zeek programs. featuresNB15.py joins with label file provided by the pcap source and calculates 12 extra features as described in (Moustafa et al 2015).
- CICFlowMeter extracts CIC features. featuresCIC.py joins with label file provided by the pcap source and adds Flow ID.

# SOURCE PCAP

AB-TRAP (bonafide) is extracted from MAWILab by filering with labeled file provided by the pcap source.
bonafide is split for processing in Argus/Zeek. Files must be joined before calculating group features.
attack is generated by simulation and each ip represents an attack category.
attack and bonafide datasets must be joined at the end.
attacks and traffic occur separatedly in time.

UNSW-NB15 label file NUSW-NB15_GT.csv is provided by the pcap source.
CIC-IDS2017 label files in GeneratedLabelledFlows.zip are provided by the pcap source.

# PCAPs
 Some packet capture files used/generated during aBFF development

* Original MAWILab pcap files extracted from: http://www.fukuda-lab.org/mawilab/

R. Fontugne, P. Borgnat, P. Abry, K. Fukuda. "MAWILab: Combining diverse anomaly detectors for automated anomaly labeling and performance benchmarking". ACM CoNEXT 2010. Philadelphia, PA. December 2010.

* CIC-BoT-IoT and CIC-ToN-IoT extracted from https://staff.itee.uq.edu.au/marius/NIDS_datasets/

M. Sarhan, S. Layeghy, and M. Portmann, An explainable machine learning-based network intrusion detection system for enabling generalisability in securing iot networks, 2021. arXiv:2104.07183 [cs.CR]

---

# MAIN FILES

> - `0_get_CIC_from_labels.py`
> - `1_Extract_Features.sh`
> - `2_Build_dataset.ipynb`
> - `3_train_model.ipynb`


## 0_get_CIC_from_labels.py 

Join CIC-IDS2017 label files as "CIC"_CIC.csv for CIC-IDS2017 case if not extracting directly from PCAP file


## 1_Extract_Features.sh [inputDirectory] 

> Generates files for later extraction of dataset features
> Requires: Argus, Zeek, cicflowmeter.py

- Argus: [sudo apt-get install argus-server] OR http://qosient.com/argus/src/argus-3.0.8.2.tar.gz
- Zeek: https://download.zeek.org/zeek-4.1.1.tar.gz
- CICFlowMeter: [sudo python3 -m pip install cicflowmeter] OR
- [git clone https://github.com/CanadianInstituteForCybersecurity/CICFlowMeter.git]

Creates inner directory per pcap file and runs Argus, Zeek and CICFlowMeter on all PCAP files in current directory.

### NOTE_1: For running MAWILab and CIC-IDS through CICFlowMeter it was necessary to force linktype to 1 [EN10MB(Ethernet)] in utils.py as: https://www.tcpdump.org/linktypes.html
### NOTE_2: Remember to edit /home/YOUR_USER/.local/lib/python3.9/site-packages/scapy/utils.py

```
import numpy
 
def __truediv__(self, other):
	type: (_Decimal) -> EDecimal
	return EDecimal(Decimal.__truediv__(self, Decimal(numpy.float64(other))))

GET ALT -> if(): else: linktype = 1
```

## 2_Build_dataset.ipynb 

> Build NB15 and CIC type datasets for each pcap
> Requires: featuresNB15.py featuresCIC.py

- SupportFils:
- featuresCIC.py: Sets labels and fixes minor formatting issues in UNBW-NB15;
- featuresNB15.py: not in use currently.

## 3_train_model.ipynb

> Train ML models for each data set.
> Requires: train_model.py

- SupportFils:
- train_model.py: Train after loading full data set or building if not found. Under or Oversamples before training if option selected.

## 4_test_model.ipynb

> Evaluate models performance in other data sets available
> Requires: evaluateBinary.py / evaluate.py

- SupportFils:
- evaluate.py: For the selected Dataset: [1] save each model's F-score table (k-split training) and [2] evaluate their F-score on other Datasets (fscore_{0}.csv)
- evaluateBinary.py: For the selected Dataset: [1] save each model's F-score table (k-split training) and [2] evaluate their full confusion matrix on other Datasets (fscore_b_{0}.csv)

## 5_XAI.ipynb  SUPERSEEDED

> Calculate SHAP values and plot feature SHAP mean value and distribution impact on classification graphs
> Requires: SHAP module

## explain.ipynb

> Calculate SHAP values and plot feature SHAP mean value and distribution impact on classification graphs
> Requires: SHAP module

- SupportFils:
- explain.py: calculates SHAP values for one model (algorithm/dataset pair) on one target dataset

## graphs.ipynb

> Build graphs for f-score

## graphs_b.ipynb

> Build graphs for different metrics using confusion matrix (including MCC)

## analyzis.ipynb

> Build feature importance graphs for each model with a personalized method

## featImportance.ipynb

> Build LaTeX tables with importance calculated as explained in analysis.ipynb