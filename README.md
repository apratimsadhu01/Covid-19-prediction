# Covid-19-prediction

In this paper, a deep convolutional neural network architecture is proposed for covid-19 prediction which is developed using TensorFlow backend and Keras deep learning library. The CNN architecture is lightweight and contains 26 layers. It is trained on 80% of 15,153 images of chest radiographs. 
The effectiveness of the model is evaluated on the basis of classification accuracy, sensitivity, specificity and f-mease. The model achieved a staggering accuracy of 98.67%. The model also achieved an average precision of 0.95, a recall of 0.94 and a f-score of 0.94. Along with this the model also achieved an AUC of 99.2. The precision, recall and f-score value of all the three classes have been presented in the paper. A comparison of accuracy of training and validation set is presented. The model achieved satisfying result in this multiclass classification problem. 
The dataset contains 15,153 chest x-ray images belonging to three classes: normal, viral pneumonia and covid-19 each of size 229x299 pixels.
Link to the dataset: https://www.kaggle.com/apratimsadhu/covid19data

This project is an end-to-end deep learning project to predict covid-19 using chest radiographs. The model is deployed through a webapp on Heroku server.A flask API is created to be deployed on Heroku cloud platform.
Link: https://covid-19-prediction-app.herokuapp.com
