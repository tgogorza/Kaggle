# Statoil/C-CORE Iceberg Classifier Challenge
## Ship or iceberg, can you decide from space?
[https://www.kaggle.com/c/statoil-iceberg-classifier-challenge#background]

### Description
Drifting icebergs present threats to navigation and activities in areas such as offshore of the East Coast of Canada.

Currently, many institutions and companies use aerial reconnaissance and shore-based support to monitor environmental conditions and assess risks from icebergs. However, in remote areas with particularly harsh weather, these methods are not feasible, and the only viable monitoring option is via satellite.

Statoil, an international energy company operating worldwide, has worked closely with companies like C-CORE. C-CORE have been using satellite data for over 30 years and have built a computer vision based surveillance system. To keep operations safe and efficient, Statoil is interested in getting a fresh new perspective on how to use machine learning to more accurately detect and discriminate against threatening icebergs as early as possible.

In this competition, you’re challenged to build an algorithm that automatically identifies if a remotely sensed target is a ship or iceberg. Improvements made will help drive the costs down for maintaining safe working conditions.

### Dataset
Training: (1604) images - 95-5 train/cv split 
Test:     (8424) images

### Implementation
The work done for this project includes:
- Pre processing of satellite images to smoothen and denoise, and also create more input channels
- Implementation of a convolutional neural network with the following architecture: 
  + 3 convolutional layers (with max pooling)
  + 3 fully connected layers (with dropout)
  + sigmoid output for binary classification

The setup, training and execution of the predictive model (CNN) was done using **TensorFlow**
