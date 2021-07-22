# KNHANES_CVD with VIF and BorutaShap

@Linux
@ray

1.Install required packages
"pip install -r requirements.txt"
(How to install ray package -> https://docs.ray.io/en/master/installation.html)

2.Prepare a dataset in CSV format(shape = (samples,columns))

3.Train & evaluation
"python3 run_ML.py --data_path ./data/sample.csv --model SVM --VIF --Boruta --train"

*data_path - path of prepared data
*model - Choose among [SVM, RandomForest,LightGBM, MLP]
*VIF - Whether to excute VIF filtering
*Boruta - Whether to excute BorutaSHAP filtering
*train - Whether to train the model.If you have already trained the model, you can omit it.

*The ray package only supports Linux. You will not be able to run this program on Windows.
