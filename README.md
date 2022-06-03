# A Fair Experimental Comparison of Neural Network Architectures for Latent Representations of Multi-Omics for Drug Response Prediction

## Install Requirements
All requirements are listed in the requirements.txt file. The usage of a virtual environment is recommended.
```shell
python3 -m pip install -r requirements.txt
```

## Download Data
Before the experiments can be performed, it is required to download the data sets from Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4036592.svg)](https://doi.org/10.5281/zenodo.4036592) and extract them into the data folder. 

## Run Experiments
### Algorithm comparison
```shell
./algorithm_comparison.sh 
```
### Ablation study
```shell
./ablation_study.sh
```