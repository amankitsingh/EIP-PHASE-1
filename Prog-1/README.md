# ASSIGNMENT 1 | EIP 4.0 |
---
##### Name: ANKIT SINGH
##### Email: ankitsingh.cool95@gmail.com
##### Github Email: https://github.com/infinityrun
---
`Background and basics of Convolutional Neural Networks`

Program Score: [0.030527586654856897, 0.9913]

`Assignment Answers`

### Convolution

In Neural Network, convolution means putting a matrix over other and calculation the weighted sum of all pixel values. One matrix will be the target image and the other will be kernel which will run of it. The standard kernel size is(3x3).
___
### Filters/Kernels 

Kernels or filters are small matrices that are used to apply some effects on the target image(matrix). The process of using these Kernels over the target images is known as Convolution. The Kernel is a square matrix generally having an odd number of rows and columns (3x3, 5x5, 7x7 ..). We make use of the different sizes of kernels in order to get different effects over our final output image like blurring, sharpening, or outlining. These are also used in ml for extracting features from an image. 
___

### Epochs :

Epochs define the number of times our algorithm/model has gone over the dataset(complete). Training a Neural network to require multiple epochs completion. 
___

### 1x1 Convolution :

convolving an image using a 1x1 kernel is called 1x1 convolution. Majorly control the depth/property of the image and pass it on to the next level.
___

### 3x3 Convolution :

This is the method of applying a 3x3 kernel over an image. By doing this only the required property of an image is retain and rest is left out. 
___
### Feature Maps :

It is the mapping of certain features over an image. When we perform multiple convolutions over an image using different kernels to get different features and these features are stacked together to become the output.In simple word's output of a kernel is called feature map.
___
### Activation Function :

In Neural Network, an activation function is defined as the output of certain channels from the given sets of input. In other words, when we give an image as input and certain channel identify some feature of its interest it gets activated.
___
### Receptive Field :

A receptive field is an area that is visible to the kernel during the sliding operations.
___
