# CIFAR-10 Image Classification Using Neural Networks

## Description

This project implements a custom convolutional neural network (CNN) architecture to classify images from the CIFAR-10 dataset. It explores various deep learning techniques such as data augmentation, dropout, batch normalization, and configurable pooling to optimize model performance. The final model achieves a test accuracy of 86.07%.

## Objectives


-Build a CNN model with a modular, block-based architecture

-Use data augmentation to reduce overfitting and improve generalization

-Apply regularization techniques like dropout and batch normalization

-Evaluate model performance across different hyperparameter configurations

## Technologies & Libraries Used

-Python

-PyTorch

-NumPy

-Matplotlib

## Neural Network Architecture

Consists of 5 blocks with varying configurations:

**Each block includes 4 convolutional layers**

-Configurable pooling layers (average or max)

-Dropout layers for regularization

**Final output block includes:**

-Batch normalization

-Mean pooling and a fully connected layer for classification


## Training Techniques

-Data augmentation: rotation & flipping improved performance significantly

-Dropout implemented to prevent overfitting

-Optimizer: Adam with learning rate 0.001

-Epochs: 40

-Hyperparameters like number of convolutional layers, kernel size, padding, pooling types were carefully tuned

## Project Structure

```

Neural-Networks-and-Deep-Learning-\

├── NNDL_code.ipynb

├── README.md

```

## Results

Best Test Accuracy: 86.07%

Training Accuracy: 92.5%

## Visualizations 

-Loss per batch (cross-entropy)

-Accuracy plots over 40 epochs

 ## Conclusion

-Model slightly overfits toward later epochs.

-Loss reduces overall with occasional spikes.




