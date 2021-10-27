# Replication Study for "Correlations Between Deep Neural Network Model Coverage Criteria and Model Quality"

## Environment Configuration
### Docker 
```bash
docker pull tensorflow/tensorflow:latest-gpu
docker run --name=covtesting --gpus all --shm-size 32G -it --mount type=bind,src=path_to_covtesting_folder,dst=/workspace tensorflow/tensorflow:latest-gpu
```
Inside the container, run the following command to install necessary libraries.
```bash
bash latest-install.sh
```

## Repo Structure:

```
- code/                                                  # Source code for Covtesting
	- Comparison of Attack Images/                   # Codes folder for comparison of attack images (Table 2, RQ1)
	- Correlation                                    # Codes folder for calculation of correlation (Figure 4, RQ1 and figure 8, RQ4)
	- Coverage VS. # Adv Examples                    # Codes folder for coverage VS. # adv examples (Figure 6, RQ2 and figure 7, RQ2)
	- Coverage and Robustness VS. Training Datasets  # Codes folder for coverage and robustness VS. training datasets (Figure 2, RQ1 and figure 3, RQ1)
	- Model Accuracy under Different Scenarios       # Codes folder for model accuracy under different scenarios (Table 3, RQ3)
	- data                                           # Reserved data folder for 'Correlation', 'Coverage VS. # Adv Examples' and 'Coverage and Robustness VS. Training Datasets'
- CONTACT                                                # Contact information of authors
- LICENSE                                                # Our code is released under the MIT license
- README                                                 # The README.md file
- esecfse2020-paper53.pdfA                               # A copy of our accepted paper
- index                                                  # The DOI of codes and data
- requirements.txt                                       # Required dependencies
```

## Data Preparation

### Data Source

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


## Experiments

Details of how to run each experiment are written as README files in each folder. Users can customize the scripts to run their own data based on provided instructions.

Results of using provided data can reproduce experimental results in the paper.






## Contact

Please fill an issue here if you have questions regarding the code.

* Shenao Yan (sy558 AT rutgers DOT edu)
* Shiqing Ma (shiqing.ma AT rutgers DOT edu)
