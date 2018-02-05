#!/bin/bash
data_csv_path=../data/adience/dataset.csv
labels_csv_path=../data/adience/gender_label.csv
train_partition=14000

echo " "
echo "<string> path of data csv file: $data_csv_path"
echo "<string> path of labels csv file: $labels_csv_path"
echo "<int> Number of rows data train: $train_partition"
echo " "

../build/test_gaussianNB $data_csv_path $labels_csv_path $train_partition