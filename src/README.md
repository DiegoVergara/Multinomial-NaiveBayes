#Adaboost Naive Bayes Multinomial C++ Intel Xeon Phi implementation#

1. Modify the path of the "CMakeList.txt" files for path cloning.

2. Imput data:

		-Download imagen database in "/data/database/aligned/" folder.

		-Modify the path of "/data/database/create_path_dataset.py".

		-Run: "python create_path_dataset.py".

3. Modify the path of /test.cpp:

		-Path of new imput dataset: Ex. "/newdataset/age_dataset_train.csv"

4. Creates the build directory and builds the application with the "makefle":

	-LBP and MultinomialNaiveBayes
	Ex.

		mkdir build

		cd /build

		cmake ../

		make

5. Run:

		./create_dataset <path to dataset.txt> <output path new datasets>; for create LBP dataset of image database

		./test ; for train and test NaiveBayesMultinomial algorithm