# Distracted Driver Detection
The dataset consists of training(22.4k) and testing images(79.7k). Here, we have made a 70-30 split on 22.4k images to get the training and testing set. The images are taken from a camera put on the dashboard of the car. Three Channel (RGB) Images are provided for the challenge. The images depict the driver being involved in certain activities. Total Size of the dataset is around 4 GB. The goal is to predict the likelihood of what the driver is doing in each picture.

The images needs to be classified into the below classes:
- c0: safe driving 
- c1: texting - right
- c2: talking on the phone - right 
- c3: texting - left
- c4: talking on the phone - left 
- c5:operating the radio
- c6: drinking 
- c7: reaching behind
- c8: hair and makeup 
- c9: talking to passenger

The code posted here involves only the machine learning implementation (and not the DL one). The details of the project are provided in ["Report.pdf"](https://github.com/gurpreet-singh-5000/Distracted_Driver_Detection/blob/main/Report.pdf). Following classification algorithms have been applied:

- SVM
- GNB
- Logistic Regression
- Nearest K-Neighbours Classifier
