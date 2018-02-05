#!/bin/bash
image_path=/home/diego/Escritorio/naive_bayes_multinomial/data/database/aligned/
dataset_path=../data/adience/dataset.txt
classification_type=gender
lbp_mapping_algorithm=u2
lbp_out_norm=true
lbp_radius=2
lbp_n_points=8
lbp_image_block=8
adaboost_n_estimators=10
adaboost_algorithm=samme
clf_alpha=1.0
clf_learning_rate=1.0
train_partition=1400
echo " "
echo "<string> path of image folder: $image_path"
echo "<string> path of 'dataset.txt': $dataset_path"
echo "<string> Type: 'age' or 'gender': $classification_type"
echo "<string> Mapping choose between: $lbp_mapping_algorithm"
echo "<bool> Output normalized histogram instead of LBP image: 'true' or 'false': $lbp_out_norm"
echo "<int> Radius: $lbp_radius"
echo "<int> Number of support points: $lbp_n_points"
echo "<int> Number of image blocks: $lbp_image_block"
echo "<int> Number of estimators: $adaboost_n_estimators"
echo "<double> alpha - Multinomial algorithm, any for Gaussian: $clf_alpha"
echo "<double> learning rate: $clf_learning_rate"
echo "<string> Adaboost algorithm: $adaboost_algorithm"
echo "<int> Number of row data train: $train_partition"
echo " "

../build/gender_classification $image_path $dataset_path $classification_type $lbp_mapping_algorithm $lbp_out_norm $lbp_radius $lbp_n_points $lbp_image_block $adaboost_n_estimators $clf_alpha $clf_learning_rate $adaboost_algorithm $train_partition