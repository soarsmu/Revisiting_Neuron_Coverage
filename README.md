# Revisiting Neuron Coverage Metrics and Quality of Deep Neural Networks

Deep neural networks (DNN) have been widely applied in modern life, including critical domains like autonomous driving, making it essential to ensure the reliability and robustness of DNN-powered systems. As an analogy to code coverage metrics for testing conventional software, researchers have proposed neuron coverage metrics and coverage-driven methods to generate DNN test cases. However, Yan et al. doubt the usefulness of existing coverage criteria in DNN testing. They show that a coverage-driven method is less effective than a gradient-based method in terms of both uncovering defects and improving model robustness.
In this paper, we conduct a replication study of the work by Yan et al. and extend the experiments for deeper analysis. A larger model and a dataset of higher resolution images are included to examine the generalizability of the results. We also extend the experiments with more test case generation techniques and adjust the process of improving model robustness to be closer to the practical life cycle of DNN development. Our experiment results confirm the conclusion from Yan et al. that coverage-driven methods are less effective than gradient-based methods. Yan et al. find that using gradient-based methods to retrain cannot repair defects uncovered by coverage-driven methods. They attribute this to the fact that the two types of methods use different perturbation strategies: gradient-based methods perform differentiable transformations while coveragedriven methods can perform additional non-differentiable transformations. We test several hypotheses and further show that even coverage-driven methods are constrained only to perform differentiable transformations, the uncovered defects still cannot be repaired by adversarial training with gradient-based methods. Thus, defensive strategies for coverage-driven methods should be further studied.

(Paper PDF)[https://arxiv.org/pdf/2201.00191.pdf]  (DOI)[]

## Environment Configuration
### Docker 
```bash
docker pull zhouyang996/covtesting
docker run --rm --name=covtesting --gpus all --shm-size 32G -it --mount type=bind,src=path_to_revisiting_neuron_coverage_folder,dst=/workspace zhouyang996/covtesting
```

Download the `adversarial-robustness-toolbox` and install necessary libraries using the following commands,
```
apt update
apt install git
git clone https://github.com/Trusted-AI/adversarial-robustness-toolbox.git
cd adversarial-robustness-toolbox
pip install .
```

## Data Preparation

```
pip install gdown

gdown https://drive.google.com/uc?id=1gUiTNIzSF_HSy6HR_Nxo8r5MkUJ-mm_C
tar -xvf data.tar.gz

gdown https://drive.google.com/uc?id=14up34H2_RVAwYmR2NNJFRI0l-FdI1d_u
tar -xvf models.tar.gz
```

## Repo Structure:

```
- Coverage and Robustness VS. Training Datasets  # Codes folder for coverage and robustness VS. training datasets (RQ1 -> Figure 1 and Figure 2, RQ2)
- Model Accuracy under Different Scenarios       # Codes folder for model accuracy under different scenarios (RQ3 and RQ4)
data                                           # data for experiment
- CONTACT                                                # Contact information of authors
- LICENSE                                                # Our code is released under the MIT license
- README                                                 # The README.md file
- requirements.txt                                       # Required dependencies
```


## Experiments

Details of how to run each experiment are written as README files in each folder. Users can customize the scripts to run their own data based on provided instructions.

Results of using provided data can reproduce experimental results in the paper.


The following is the original data from [original paper](https://github.com/RU-System-Software-and-Security/CovTesting), the structure of which is a bit unclear. We don't use them.


### The original Data Source

* Download data and codes through the DOI link: <a href="https://doi.org/10.5281/zenodo.3908793"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3908793.svg" alt="DOI"></a>
* Google Drive mirror: https://drive.google.com/drive/folders/16w93LPkaF0AP9QxIty9Y6ipU-N4cbPUd?usp=sharing

You will get two zip files: 'all-data.zip' (12.2 G, the data file) and 'DNNTesting/CovTesting-v1.1.zip' (5.2 M, the codes file). Please unzip the codes and data files to get the codes and data for experiments. After unzipping 'all-data.zip', you will get three zip files named 'data.zip' (4.1 G), 'Table 2 data.zip' (3.5 G) and 'Table 3 data.zip' (4.6 G).

Alterantively, use the following command to download:
```
# Download the 'data' folder, make sure that you are under the root folder.
gdown https://drive.google.com/uc?id=1bClp6T9VuvTzSwspVf_zRkZptI31F6XI
# Decompress it
unzip data.zip

## Download 'Table 2 data.zip' 
cd Comparison\ of\ Attack\ Images/
gdown https://drive.google.com/uc?id=1DZK2gY6Gz991FS_5mXtLnsx3ZiVhUNMY
# Decompress it
unzip Table\ 2\ data.zip 
mv Table\ 2\ data/data/ data/
rm -r Table\ 2\ data

## Download 'Table 3 data.zip'
cd Model\ Accuracy\ under\ Different\ Scenarios/
gdown https://drive.google.com/uc?id=18MN6HNT-9DR6cML6r8mInPb4GC7drRBc
unzip unzip Table\ 3\ data.zip
mv Table\ 3\ data/data/ data/
mv Table\ 3\ data/new_model/ new_model/
rm -r Table\ 3\ data
```


* Unzip 'data.zip' and get three folders: 'cifar_data', 'mnist_data' and 'svhn_data'. Please put these three folders under the ['data' folder](https://github.com/DNNTesting/CovTesting/tree/master/data) in the main folder. 
* Unzip 'Table 2 data.zip' and get three folders: 'cifar_data', 'mnist_data' and 'svhn_data'. Please put these three folders under the ['data' folder](https://github.com/DNNTesting/CovTesting/tree/master/Comparison%20of%20Attack%20Images/data) under the ['Comparison of Attack Images' folder](https://github.com/DNNTesting/CovTesting/tree/master/Comparison%20of%20Attack%20Images). 
* Unzip 'Table 3 data.zip' and get two folders: 'data' and 'new_model'. Please merge these two folders into the ['data' folder](https://github.com/DNNTesting/CovTesting/tree/master/Model%20Accuracy%20under%20Different%20Scenarios/data) and the ['new_model' folder](https://github.com/DNNTesting/CovTesting/tree/master/Model%20Accuracy%20under%20Different%20Scenarios/new_model) under the ['Model Accuracy under Different Scenarios' folder](https://github.com/DNNTesting/CovTesting/tree/master/Model%20Accuracy%20under%20Different%20Scenarios), respectively. 