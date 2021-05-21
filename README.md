---

![image](https://user-images.githubusercontent.com/60827480/119011524-70488a80-b995-11eb-8ddd-55f7c9363572.png)

---

This project outcome belongs to **[Arlene Postrado](https://github.com/arlene14ko)**, **[Daniel Mendoza](https://github.com/danielmendoza4213)**, **[Louan Mastrogiovanni](https://github.com/Louan-M)** and **[Martin Makyeme](https://github.com/makyeme)** who are currently Junior Data Scientists/AI Operators in making at BeCode's Theano 2.27 promotion.

- Repository: `yoga_pose`
- Type of challenge: `Learning`
- Level: `Junior Data Scientist`
- Duration: `5 days`
- Deadline : `19/05/21 5:00 PM`
- Team challenge: `Group Project`
- Deployment Strategy: `Github page and Flask`
- Promotion : `AI Theano 2`
- Coding Bootcamp: `Becode Artificial Intelligence (AI) Bootcamp`

---

## Mission Objectives

- Be able to work with computer vision libraries, and techniques for tracking poses on images on videos
- Be able to explore the pre-trained models for pose tracking on live and streaming media
- Be able to predict Yoga poses in real time
- Be able to use Neural Networks and Machine Learning models
- Be able to deploy the models for end customers

---

## The Mission

YogaLive is a platform that provides yoga classes virtually. YogaLive is looking for a solution to improve the quality of their services, the main challenge they face is to analyse how students are performing the different poses in a normal yoga class. Being virtual and live classes, it can be difficult for teachers and students to know how well the poses are being performed, this can be due to both the number of students and technical issues such as the quality of the video

The company has provided us with sample images of different poses performed by one of their teachers, in order to obtain a model that can recognise the type of pose, estimate the duration of each pose and how many times this pose has been performed. If possible also a model to get a dayly, weekly or monthly student report.

Update: Data has been increase with google images to test models

---

### Table of Contents

- Packages Requirements
- Repository
- Visual
- Pending things to do
- Collaboration

---

## About the Repository

This is a repository is about developing a model that is able to predict a yoga pose using pretrained Convolutional Neural Networks and the computer vision libraries and techniques.

This project is currently deployed locally, if you wanted to try to run this on your own and you dont have a GPU on your computer, you can use [Google Colab](https://colab.research.google.com/) as it needs a lot of computing power.

**Here is a sample dashboard:**!

![image](https://user-images.githubusercontent.com/60827480/119103353-5baed580-ba1b-11eb-9537-4cdc7afc5376.png)


---

### Repository

This has 2 versions:

1. You can run this program locally using the **app.py**
2. If you dont have a GPU, you can run the **YogaLive_app_py.ipynb** notebook using [Google Colab](https://colab.research.google.com/).

**0. Exploration Phase folder**

- This is where the exploration about the domain is done.
- This currently has 1 notebook namely:

  1. **landmark_from_images.ipynb**
     - this is where we explored how to get the landmarks in an image using mediapipe
  2. **real time prediction.ipynb**
     - this is where we explored how to get the landmarks with the test video

**1. MACHINE LEARNING folder**

1. **Creating the training dataset.ipynb**

   - this is the notebook where we first created our dataset using MediaPipe and getting those landmarks, saving those landmarks in a csv file and creating a Machine Learning Model using that csv file

2. **SVM.ipynb**

   - this is the notebook where we created the Machine Learning Model

3. **yoga_poses_model.pkl**
   - this is the saved Machine Learning model using pickle and this used the MediaPipe landmarks

**2. NEURAL NETWORK folder**

1. **CNN_Model.py**

   - this is where our CNN model was created and was later used as our model to get the pose analysis

2. **VGG16.ipynb**

   - this is the pretrained CNN model which we used to create our model

3. **final_model folder**
   - this folder contains the final model of our CNN, but we did not upload it because it is too heavy. If you want to know more, dont hesitate to contact us!

**.gitignore**

- is a txt file used for specifying which files/folders does git needs to ignore so it will not upload to github
- This is used because our dataset cannot be uploaded as it is confidential and also our CNN final model was 1.25GB which the Github cannot handle as of the moment
- If you want to have access to our final model, please dont hesitate to send us a message! :)

**static folder**

- this is where the logo and favicon is saved

**templates folder**

- this is where all the html templates are saved
- this has 3 html files namely:

  1.  **layout.html**

      - this is where all the imports for the html is located
      - this is also where the NavBar and the Footer is saved so it will be shown on all pages

  2.  **analyze.html**

      - this is where the the user needs to input the details to be able to analyze the poses
      - it has a tab depending on the user needs:

              - Live tab - allows the user to use live stream video to analyze the pose
              - Upload tab - allows the user to upload a video file to analyzethe pose

  3.  **live_feed.html**
      - this is the page where you will see the video with the analyzed poses and the details of the pose

**app.py**

- You can run this repository on your local computer using this python file
- Flask app containing all the functions for the program to run in flask
- Routes include but not limited to: - route that will ask the user input - route that will show the live feed/stream and predicting the poses
- Functions include but not limited to: - function that get the video, extract frames from that video and predict the the poses - function that calculate the angles

**yogaLive_app_py.ipynb**

- You can run this notebook using [Google Colab](https://colab.research.google.com/)'s GPU
- This notebook contains all the functions for the program to run in Google Colab, it also includes code for you to connect Google Colab with your drive to properly run the code
- Routes include but not limited to: - route that will ask the user input - route that will show the live feed/stream and predicting the poses
- Functions include but not limited to: - function that get the video, extract frames from that video and predict the the poses - function that calculate the angles

**README.md**

- has all the necessary information regarding the project
- It would be highly recommended to read all the information in the README file.

---

## Libraries Used For This Project

**OpenCV** https://opencv.org/

- OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library.
- In this project, OpenCV is used to read the videos and get the frames from each image

**Keras** https://flask.palletsprojects.com/en/1.1.x/

- Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages.
- In this project, Keras is used to create the model easier

**MediaPipe** https://mediapipe.dev/

- MediaPipe offers open source cross-platform, customizable ML solutions for live and streaming media.
- In this project, MediaPipe pose is used to extract the landmarks to use it for training our SVM model and also afterwards to get the landmarks to calculate the angles

**Flask** https://flask.palletsprojects.com/en/1.1.x/

- Flask is a micro web framework written in Python. It is classified as a microframework because it does not require particular tools or libraries
- In this project, flask is used to create the web dashboard application

**Pandas** https://pypi.org/project/pandas/

- Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.
- In this project, pandas is used convert a list to create a dataframe and convert that dataframe to a csv

---

## Clone / Fork This Repository

- This project is open to collaborations as well as forking or cloning for further development. If you wish to clone/fork this repository, you can just click on the repository, then click the Clone/Fork button and follow the instructions.

## P E N D I N G . . .

- We believe that with a bigger dataset, accuracy of the predictions models can be imporved.

![Thank you](https://static.euronews.com/articles/320895/560x315_320895.jpg?1452514624)

### Thank you for reading. Have fun with the code! 🤗
