#!/bin/bash
data_csv_path=../data/adience/dataset.csv
labels_csv_path=../data/adience/gender_label.csv
n_estimators=10
algorithm=samme
alpha=1.0
learning_rate=1.0
train_partition=14000

echo " "
echo "<string> path of data csv file: $data_csv_path"
echo "<string> path of labels csv file: $labels_csv_path"
echo "<int> Number of estimators: $n_estimators"
echo "<string> Adaboost algorithm: $algorithm"
echo "<double> alpha - Multinomial algorithm, any for Gaussian: $alpha"
echo "<double> learning rate: $learning_rate"
echo "<int> Number of rows data train: $train_partition"
echo " "

../build/test_adaboost $data_csv_path $labels_csv_path $n_estimators $algorithm $alpha $learning_rate $train_partition