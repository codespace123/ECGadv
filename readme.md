## ECGadv: Generating Adversarial Electrocardiogram to Misguide Arrhythmia Classification System

This code is for paper "ECGadv: Generating Adversarial Electrocardiogram to Misguide Arrhythmia Classification System".

### Installing
* Python 3.6.5
* Tensorflow 1.8.0
* Keras 2.2.0
* [Cleverhans 2.0.0](https://github.com/tensorflow/cleverhans) 

This can be done on Linux using
```
pip install https://github.com/mind/wheels/releases/download/tf1.8-cpu/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl
pip install keras
pip install cleverhans
```

### Prerequisites

Download the following items:

* [Dataset](https://physionet.org/challenge/2017/training2017.zip) 
* [Revised labels for the dataset](https://physionet.org/challenge/2017/REFERENCE-v3.csv)
* [Targeted model](https://github.com/fernandoandreotti/cinc-challenge2017/blob/master/deeplearn-approach/ResNet_30s_34lay_16conv.hdf5)

```
wget https://physionet.org/challenge/2017/training2017.zip
unzip training2017.zip
wget https://physionet.org/challenge/2017/REFERENCE-v3.csv
wget https://github.com/fernandoandreotti/cinc-challenge2017/blob/master/deeplearn-approach/ResNet_30s_34lay_16conv.hdf5
```

### File Description

PrepareAttackDataset.py: apply the [targeted model](https://github.com/fernandoandreotti/cinc-challenge2017/blob/master/deeplearn-approach/ResNet_30s_34lay_16conv.hdf5) on the [dataset](https://physionet.org/challenge/2017/training2017.zip), record the ones predicted the same as in [label file](https://physionet.org/challenge/2017/REFERENCE-v3.csv), i.e., correct prediction. 

prediction_correct.csv: output file of [PrepareAttackDataset.py](PrepareAttackDataset.py), includes all correct predictions. 

data\_select\_A.csv, data\_select\_N.csv, data\_select\_O.csv, data\_select\_i.csv: output file of [PrepareAttackDataset.py](PrepareAttackDataset.py), includes the correct predictions of class A, N, O, ~ respectively. 


#### For attacks against cloud deployment model: 

* myattacks\_l2.py & myattacks\_tf\_l2.py: Attack func with similarity metric d<sub>l2
* myattacks\_diff.py & myattacks\_tf\_diff.py: Attack func with similarity metric d<sub>smooth
* myattacks\_diffl2.py & myattacks\_tf\_diffl2.py: Attack func with similarity metric d<sub>smooth,l2

* cloud\_eval\_l2.py: Generate attack perturbation by calling [myattacks\_l2.py](myattacks\_l2.py), and save it in "./cloud\_model/l2\_eval/"
* cloud\_eval\_diff.py: Generate attack perturbation by calling [myattacks\_diff.py](myattacks\_diff.py), and save it in "./cloud\_model/smooth\_eval/"
* cloud\_eval\_diffl2.py: Generate attack perturbation by calling [myattacks\_diffl2.py](myattacks\_diffl2.py), and save it in "./cloud\_model/l2smooth\_0\_01\_eval/"

Example run:

```
python attack_file index_file start_idx end_idx
```
* *attack_file*: can be "cloud\_eval\_l2.py", "cloud\_eval\_diff.py", "cloud\_eval\_diffl2.py"
* *index_file*: can be "data\_select\_A.csv", "data\_select\_N.csv", "data\_select\_O.csv", "data\_select\_i.csv"
* *start_idx*: integer, at least 1
* *end_idx*: integer, not included 

./cloud\_model/metric_compare.py: given *idx*, *TRUTH* and *TARGET*, plot a figure including the original sample and three adversarial ones. 

#### For attacks against local deployment model: 

* LDM_EOT.py & LDM_EOT_tf.py: Attack func with EOT
* LDM_Attack.py: Generate attack perturbation for Local Deployment Model and save it in "./output/$GroundTruth/" 

Example run:
```
python LDM_Attack.py ID Target Window_Size
```

* *ID* - sample ID for attack 
* *Target* - target class
* *Window_Size* - perturbation window size

*LDM_UniversalEval.py: Test perturbation generated from LDM_Attack.py. The program will load the perturbation in "./output/$GroundTruth/ and test it via adding it to all the samples in data_select_?.csv with the targeted class. 

Example run: 
```
python LDM_UniversalEval.py ID Target Window_Size
```

* *ID* - The ID of samples that generate the perturbation.
* *Target* - target class. 0,1,2,3 represents A, N, O, ~ respectively
* *Window_Size* - perturbation window size. The integer value is better to divide the length of the origin sample which is 9000. Because we haven't test other conditions.

To demostrate the universality of the attack, the program will test all the samples in data_select_$Target.csv that belong to *Target*.

#### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
