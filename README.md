# Behaviorial Cloning Project
** Jianguo Zhang, May 03, 2017**

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


The project includes five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.






[//]: # (Image References)

[image1]: ./examples/nVidia_model.png "Model Visualization"
[image9]: https://github.com/JianguoZhang1994/LeNet-written-by-tensorflow/blob/master/lenet.png "Grayscaling"
[image2]: ./examples/center_2017_05_04_00_08_13_140.jpg "Center Image"
[image3]: ./examples/center_2017_05_04_00_08_13_140.jpg "Center Image"
[image4]: ./examples/left_2017_05_04_00_08_13_140.jpg "Left Image"
[image5]: ./examples/right_2017_05_04_00_08_13_140.jpg "Right Image"
[image6]: ./examples/normal_image.jpg "Normal Image"
[image7]: ./examples/flipped_image.jpg "Flipped Image"
[image8]: ./examples/multiple_cameras.jpg "Multiple Cameras"


#### 1. Project includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 2. Project code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

For my model, I modify the [Nvidia End to End learning Architecture](https://arxiv.org/pdf/1604.07316.pdf), where my image shape is (160, 320, 3). The model consists of 5 convolution neural layers with 5x5 and 3x3 filter sizes and depths between 24 and 64, there are also 3 full connected layer, the output of final layer is 1(model.py lines 76-88).

The model includes ELU layers to introduce nonlinearity, the data is normalized and cropping in the model using a Keras lambda layer and cropping method(code line 77-78). we use the activation function ELU instead of RELU, The benefits of ELU's over ReLU's have been published in the [FAST AND ACCURATE DEEP NETWORK LEARNING BY EXPONENTIAL LINEAR UNITS ELUS](https://arxiv.org/pdf/1511.07289v1.pdf).

#### 2. Attempts to reduce overfitting in the model

I set the epoches as 2(model.py code line 94) and batch size as 32(code line 66-67). The model doesn't contain dropout layers here. But if I increase the number of epoches and batch size, I may need to contain dropout layer in order to reduce overfitting(code line 87). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 20-60) as each image maybe flipped and changed steering angle. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 90).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and flipped images.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the [LeNet](https://github.com/JianguoZhang1994/LeNet-written-by-tensorflow), I thought this model might be appropriate because it has been used successfully in many image related tasks. When I test the network, the car easily drive outside of roads, I think the failure reason are LeNet is not enough deep, it only consists of two convolutional layers, besides, the filters size maybe also not enough to deal with a (160, 320,3) size image.  

<div align=center>
<img src="https://github.com/JianguoZhang1994/LeNet-written-by-tensorflow/blob/master/lenet.png?raw=true" width="600px">
</div>

#### 2. Final Model Architecture

The final model architecture (model.py lines 76-88) consisted of a convolution neural network with the following layers and layer sizes shows as the image, where the final layer output is 1. 

Here is a visualization of the architecture (note: the final layer is 1)

<div align=center>
<img src="./examples/model_visualization.jpg?raw=true" width="600px">
</div>

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

<div align=center>

![alt text][image2]
</div>

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to come back to center when drivinh toward outside of roads. These images show what a recovery looks like:

<div align=center>

<img src="./examples/left_2017_05_04_00_08_13_140.jpg?raw=true" width="290" height="160" />
<img src="./examples/center_2017_05_04_00_08_13_140.jpg?raw=true" width="290" height="160" />
<img src="./examples/right_2017_05_04_00_08_13_140.jpg?raw=true" width="290" height="160" />
</div>

Then I repeated this process on track two in order to get more data points.

To augment the data sat, Firstly, I use multiple cameras, I randomly choose one image among ['center', 'left', 'right'] for each line(code line 37-43), I add a correction to 'left' image and substract a correction to 'right'image, where the correction value need to be tuned. I also randomly flipped images and angles(model.py code line 49-54) thinking that this would make the data more comprehensive and unbiased. For example, here is an image that has then been flipped:

<div align=center>

<img src="./examples/multiple_cameras.jpg?raw=true" width="600px">

</div>

<div align=center>

![alt text][image6]
![alt text][image7]

</div>



With flipped images, it looks like the car drive in an opposite direction.

After the collection process, I had 12673 number of data points. each data point consists of center, left and right image. I then preprocessed this data by randomly choose a image from ['center', 'left', 'right'] images for each image and randomly flip images, note that each time the output are same for each batch size. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as the batch size is small. I used an adam optimizer so that manually training the learning rate wasn't necessary.



### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp when the image image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).
