Dataset and gloveembeddings are already downloaded 
preprocessing is already done , so commented out that part .uncomment in train.py if needed

create a conda environment usiing below command, all the neccessary libraries will be loaded
conda env create -f environment.yml

to train the model --> python train.py

to test the model--> python test.py  --> output file target_model_results will be generaated 

There is already one pre trained model so model can be directly tested

However if u chose to train the model it may take up to 6 hrs if cuda is not enabled 

if u wish to change any hyperparameters , it can be done in util/constants.py file