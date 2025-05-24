#Hybrid FrameWork for Viral Genome Classification Using FCGR Image with CNNs and XGBoost
Author: Sunil Kumar, Tejas S Gowda, Mohammed Khalid Ahmed, Kodi Satya Sai Prakash

# CNN Training

- Add all the virus names and their accession ids in the 'data_generator/accession_ids.json' file.
- Run 'fetch_sequence.ipynb'.
- Run 'FCGR_generator.ipynb'. This is CPU intensive and takes around 2 hrs to generate 1.2 lakh images.
- After running the above scripts a new 'data' folder will be generated which can readily be using for training using pytorch.

- Change which GPU to train on using 'device = 'cuda:0''. You can see which GPU is free by running 'nvidia-smi' command.
- List of pretrained models are available at 'Torchvision pretrained' models site.
- We have already included some model.ipynb files. A new model.ipynb file can be generated similarly for a different model.
- It is recommended to generate new model.ipynb files for each model as the model blocks can differ and hyperparametes vary.
  For example AlexNet has 'feature' and 'classifier' block but it may not be the case for all models.
- Usually we freeze all the blocks except the classifier blocks. Without this some complex models take days for training.
- You can know the model architecture just by printing the 'model' variable.
- Each model (freezed except the classifier block) can take an average of 6 hrs for training for 100 epochs.
- As you train the model all the metrics are automatically tracked and continuously updated in the 'model_saves/model.txt' file.
  So you can catch any potential underfitting or overfitting.
- If you see any anomaly then you can easily adjust the hyperparameters, scheduler and optimizer.
- After the model is completely trained the model weights are saved in 'model_saves/model.pth'.
- These '.pth' files can be used for further analysis like feature extraction, classification...

# LASSO XGBoost Training

- The required trained model weights '.pth' file should be placed inside 'model_saves' folder.
- Change model_loader('model_name') to the file name in the 'model_saves' folder.
- Pls note that H2Os LASSO Regression is very CPU intensive for large number of features. Since this is a
  parallel implementation it uses all the cores and will slow down the system massively.
