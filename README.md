# DatabaseProject
Seq2SQL Project

pre processed Dataset and glove embeddings are compressed and hosted in below drive link
https://drive.google.com/file/d/19bS7whOzOEEDFMpQ-0v_OVGdxbZVBKpy/view?usp=drive_link

1.Download and extract above file in project folder.

recommeded to use conda for ease of setting up as we are sharing our environment 
create a conda environment usiing below command, all the neccessary libraries will be loaded
2.conda env create -f environment.yml

3.preprocessing is already done , so commented out that part .uncomment in train.py if needed

4.if u wish to change any hyperparameters , it can be done in util/constants.py file

5.to train the model --> python train.py
There is already one pre trained model so this step can be skipped and  model can be directly tested

6.to test the model--> python test.py 
output file target_model_results will be generaated 
already file exists from previous iteration , delete it before running


