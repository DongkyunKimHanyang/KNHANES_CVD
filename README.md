# Classification with VIF and BorutaShap


## Environments

- [Ubuntu 20.04]
- [Python>=3.6] 
- [Scikit-learn]
- [ray>=1.2]

## Installation
Install python packages to run.  
More details about installing ray package -> https://docs.ray.io/en/master/installation.html.  

```sh
$pip3 install -r requirements.txt
```
## Prepare data
Data source : https://knhanes.kdca.go.kr/knhanes/main.do.  
The data should be in CSV format and the dimensions should be (samples, columns).  
Then write the types of your variables in variables.csv (numeric or binary)  [variable.csv](./data/variables.csv)  

## Run

```sh
$ray start --head --port=6379
To connect to this Ray runtime from another node, run
    ray start --address='your address' --redis-password='your password'
$ray start --address='your address' --redis-password='your password'
$python3 run_ML.py --label Cardio --data_path ./data/sample.csv --model SVM --VIF --Boruta --train
```
`--label`: Name of target column.  
`--data_path`: Path of prepared CSV file.  
`--model`: Choose a machine learning model among ['SVM','RandomForest','LightGBM','MLP'].  
`--VIF`: Boolean argument for VIF filtering.  
`--Boruta`: Boolean argument for BorutaSHAP filtering.  
`--train`: Whether to train the model. True: training & evaluation, False: only evaluation 


Training and evaluation results are automatically created in the ./result directory.

*The ray package only supports Linux. You will not be able to run this program on Windows.  

We modified BorutaSHAP of https://github.com/Ekeany/Boruta-Shap with parallel computation using ray package.
