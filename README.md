# Database Project - Seq2SQL

pre processed Dataset and glove embeddings are compressed and hosted in below drive link https://drive.google.com/file/d/19bS7whOzOEEDFMpQ-0v_OVGdxbZVBKpy/view?usp=drive_link

1. Download and extract above file in project folder.
   recommeded to use conda for ease of setting up as we are sharing our environment create a conda environment usiing below command, all the neccessary libraries will be loaded

3. Run `conda env create -f environment.yml`

4. Since the preprocessing of the dataset is already completed, you may comment out `.uncomment` in `train.py`.

5. If u wish to change any hyperparameters , you can do so in the `util/constants.py` file

6. To train the model, run `python train.py` in conda. Whole process of training may take upwards of 6 hrs. There is already one pre trained model so this step can be skipped and model can be directly tested

7. To test the model, run `python test.py` in conda output file target_model_results will be generaated already file exists from previous iteration , delete it before running


