# InfoTS

This repository contains the implementation for the paper *Information-Aware Time Series Meta-Contrastive Learning*. We implement based on the TS2vec repository(https://github.com/yuezhihan/ts2vec). Thanks for their public avilable code.

## Requirements

The recommended requirements are specified as follows:
* Python 3.7.7
* scipy==1.6.1
* torch==1.7.1
* numpy==1.19.2
* pandas==1.0.1
* scikit_learn==0.24.1


## Data

The datasets can be obtained and put into `datasets/` folder in the following way:

* [128 UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018) should be put into `datasets/UCR/` so that each data file can be located by `datasets/UCR/<dataset_name>/<dataset_name>_*.csv`.
* [30 UEA datasets](http://www.timeseriesclassification.com) should be put into `datasets/UEA/` so that each data file can be located by `datasets/UEA/<dataset_name>/<dataset_name>_*.arff`.
* [3 ETT datasets](https://github.com/zhouhaoyi/ETDataset) should be placed at `datasets/forecast/ETTh1.csv`, `datasets/forecast/ETTh2.csv` and `datasets/forecast/ETTm1.csv`.
* [Electricity dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014) should be resampled into hourly data of 321 clients over the last 3 years and placed at `datasets/forecast/electricity.csv`.


## Usage

Full code will be available upon acceptance.



## Code Example

Example.ipynb