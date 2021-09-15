# DNA sequence Classification

In this repository, given a list of DNA sequences bound to an unknown Transcription Factor (TF), we would like to gain insight into its binding preferences and build a predictor that helps identifing novel sequences that would bind to it.

## Installation
To prepare your environment, run the following command:
```
conda env create -f env.yaml
```

## Train
To train a model, use the following command:
```
python run.py --mode=tain --data_path=./train.csv
```

where `train.csv` is assumed to be a csv file containing bound sequences. In the preprocessing step, the code would generate control sequences by modifying bound sequences. 
You may add more command line argument to change the hyper-parameters and output path.

## Test
In order to test a trained model, run the following command:
```
python run.py --mode=eval --data_path=./test.csv --model_path=./model.pt
```
where `test.csv` is supposed to be comma-separated csv file containing sequences and labels and `model.pt` is the checkpoint of the trained model.
