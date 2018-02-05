import pandas as pd
import numpy as np

fold0 = pd.read_csv("fold0.csv", sep = ";", dtype = str)
fold1 = pd.read_csv("fold1.csv", sep = ";", dtype = str)
fold2 = pd.read_csv("fold2.csv", sep = ";", dtype = str)
fold3 = pd.read_csv("fold3.csv", sep = ";", dtype = str)
fold4 = pd.read_csv("fold4.csv", sep = ";", dtype = str)

frames = [fold0, fold1, fold2, fold3, fold4]

folds = pd.concat(frames)

del fold0
del fold1
del fold2
del fold3
del fold4

data = folds.dropna()

del folds

data['link'] = data['user_id'].str.cat(data['efect'], sep='/')
data['link2'] = data['face_id'].str.cat(data['original_image'], sep='.')
data['link3'] = data['link'].str.cat(data['link2'], sep='.')

del data['link'];
del data['link2'];
del data['original_image'];

#path = '/home/sergio/code/cpp/naive_bayes_multinomial/data/adience/aligned/'
path = ''

data['link'] = path+data['link3'];

del data['link3'];
del  data['user_id'];
del data['face_id'];
del data['efect'];

data[['link', 'age', 'gender']].to_csv("dataset.txt", sep = ';', header = False, index = False)
