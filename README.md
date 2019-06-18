# Machine Learning Nanodegree
## Visual Question Answering system
## Author
### Abishek chiffon

### Files
* vqa.py- This files contains all the code to dowload and organise the data
* VQA1_Attention_model.ipynb - This file contains the Attention model
* VQA1_LSTM_Without_elmo.ipynb - This file contains the LSTM without elmo
* VQA1_elmo_final_notebook_with_prediction_errors - This file is the same as VQA_final except in this the prediction error is demonstrated in the prediction section
* VQA_final- This is the final model notebook that contains the lstm implementaion with elmo.


### Folders
* graphs - This folder contains the graph generated in terms of epochs
* models - This folder contains the model for the model. This will contain only one model, which has an validation accuracy of 60%
* modules - This folder will contain elmo modules . "https://tfhub.dev/google/elmo/2" can also be used in the elmo embedding layer code.
* train/train2014 - train images
* train/annotation and question files

### Usage 
1. Download the train and test data from the vqa download page.
[train images](http://images.cocodataset.org/zips/train2014.zip)
[question](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip)
[answer](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip)
Place them in the train folder 
2. To extract features from the train data.
Execute the VQA1_elmo_final_notebook in order
Uncomment the lines under  "Image conversion to .npy files" section
```python
feel free to change the batch_size according to your system configuration

image_dataset = tf.data.Dataset.from_tensor_slices(
                                image_name_vector[:n_train_samples]).map(load_image_1).batch(16)

for img, path in tqdm(image_dataset):
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features, 
                              (batch_features.shape[0], -1, batch_features.shape[3]))

    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())
```

3. Pass the models in the models folder to the load_weights function of execution step to execute th e model 
```python
model = build_model()
model.load_weights("model_name")
model.fit_generator()
```
4. Run the execude code section to fit the model to the data
5. Run the evaluation step to evaluate the model.

###GITHUB
The performance of the model will be continuted to be improved and all the files will be available in git hub repository.
[Abishek](https://github.com/abishekchiff/udacity_vqa)

