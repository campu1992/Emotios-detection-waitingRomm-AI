# Emotios-detection-waitingRomm-AI

**Project README**

## Emotion Recognition from Video

This project aims to recognize emotions from a video stream using computer vision and deep learning techniques. The process involves detecting faces in each frame of the video and predicting the corresponding emotions for each detected face.

### Libraries Used

The project utilizes the following libraries:

- **OpenCV (cv2)**: Used for reading video frames, face detection, and image preprocessing.
- **NumPy**: Utilized for array manipulation and mathematical operations.
- **TensorFlow**: TensorFlow and its Keras API are used for building, training, and loading the convolutional neural network (CNN) model for emotion classification.
- **CIFAR-10 Dataset**: TensorFlow's built-in dataset for training and testing the CNN model.
  
### Steps Involved

1. **Loading and Preprocessing Data**:
   - The CIFAR-10 dataset is loaded, which contains labeled images for training and testing.
   - Images are normalized to values between 0 and 1.

2. **Defining CNN Architecture**:
   - A Convolutional Neural Network (CNN) model is defined using TensorFlow's Keras API.
   - The model consists of convolutional layers followed by max-pooling layers, flattening, and dense layers with ReLU and softmax activations.

3. **Compiling and Training the Model**:
   - The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function.
   - Training data is fed into the model for a specified number of epochs to learn patterns and relationships.

4. **Evaluating the Model**:
   - The trained model is evaluated on the test dataset to assess its performance in terms of accuracy.

5. **Saving the Trained Model**:
   - Once trained, the model is saved for future use in emotion classification tasks.

6. **Emotion Classification from Video**:
   - The saved model for emotion classification is loaded.
   - A video file is opened for analysis using OpenCV.
   - Frames are processed sequentially:
     - Faces are detected using OpenCV's face detection algorithms.
     - Detected faces are preprocessed to match the input requirements of the emotion classification model.
     - Emotions are predicted for each detected face using the loaded classification model.
     - Predicted emotions are overlaid on the corresponding face regions in the video frame.
   - The processed frames with overlaid emotions are displayed in a window.
   - Pressing 'q' terminates the video analysis.

7. **Cleaning Up**:
   - Upon exiting the analysis loop, the video window is closed, and resources are released.

