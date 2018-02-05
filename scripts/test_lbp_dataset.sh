#!/bin/bash
#image_path=../data/adience/aligned/
image_path=/home/diego/Escritorio/naive_bayes_multinomial/data/database/aligned/
#fn_csv=../data/adience/dataset.txt
fn_csv=../data/adience/dataset.txt
class_type=gender
mapping=u2
normalizeHist=true
rad=2
pts=8
subi=8
output=../data/adience/

echo " "
echo "<string> path of image folder: $image_path"
echo "<string> path of 'dataset.txt': $fn_csv"
echo "<string> Type: 'age' or 'gender': $class_type"
echo "<string> Mapping choose between: $mapping"
echo "<bool> Output normalized histogram instead of LBP image: 'true' or 'false': $normalizeHist"
echo "<int> Radius: $rad"
echo "<int> Number of support points: $pts"
echo "<int> Number of image blocks: $subi"
echo "<string> Output path: $output"
echo " "

../build/test_lbp_dataset $image_path $fn_csv $class_type $mapping $normalizeHist $rad $pts $subi $output