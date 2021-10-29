## Dependencies

1. Create a new conda virtual environment (python 3.6)
2. run `pip install -r requirements.txt`
3. install the newest IBM adversarial-robustness-toolbox:

First clone the repository, `git clone https://github.com/IBM/adversarial-robustness-toolbox`

Then, `pip install . `

## Data Preparation
Unzip the two compressed files: `data.zip` and `models.zip` into the root folder as `./data` and `./`



## Running
Since we have multiple models to train, we want to utilize multiple GPUs to train.
Please go to `./Correlation` folder, and then run `python run_adv_train.py`
It will train multiple models in parallel. 

These folders will be stored under `./models` folder.

