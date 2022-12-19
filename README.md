# Out-Of-Distribution Detection for Event Sequences

**Authors:** Belousov Nikita, Ulyanova Maria

## Problem

In this project we consider a challenging problem of anomaly detection algorithms. We will concentrate on distinguishing abnormal financial data for each unique user.

Our main approach will be to use autoencoding techniques for data reconstruction. As a result we hope to get abnormal loss growth for the anomaly samples.

## Requirements

All the requirements are listed in `requirements.txt`

For install all packages run.

```
pip install -r requirements.txt
```

## Data

You can find all of the necessary data in [here](https://disk.yandex.ru/d/pYijj1fHonHRSw).

## Experiments

Logs, model and results you can find [here](/lightning_logs).

## Results

As the result, we trained two autoencoder models and meta-classifier for distinguishing anomaly transactions. Results we've got tell us about possibility of out method. So, as our future work with this project we will be moving rapidly toward the GAN methods.


