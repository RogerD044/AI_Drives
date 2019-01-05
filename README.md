# AI_Drives

A virtual self driving car simulation implemented in TensorFlow. 

## Downloading Dataset

You can download the dataset from [here](https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view)
```
Size ~ 2.2 GB (45,000 images)
```

## Getting Started

Download the dataset and extract into the repository folder.

The dataset will contain approx. 45,000 images along with a file data.txt which contains the steering angle for the images captured by the cars front camera.

The 45,000 images resemble 25 mins, first 80% of which is kept for training the model and the performance is tested on the last 20%.

### Folders
- logs : saves the logs to tensorboard
- save : saves the trained models and checkpoints

### Files
- Self_driving_car_Analysis.ipynb : Explores the dataset
- load_data.py : To load the data from the dataset
- model.py : Creates a CNN model based on this [paper](https://arxiv.org/abs/1704.03952)
- train.py : Trains and saves the model
- test_data.py : Runs the trained model on the test data
- Training CNN.png : Structure of the CNN model with slight modifications
- steering.jpg : Steering wheel for better visualization during testing

