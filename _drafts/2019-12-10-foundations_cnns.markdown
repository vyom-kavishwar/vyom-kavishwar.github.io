#### (1) *LeNet5 (LeCun et al, 1989)*
<img src="/assets/LeNet_5.png" alt="fig_3" width="700"/>

Way back in 1989 (much before I was even born!), Yann LeCun et al developed LeNet - the first convolutional neural network [3].

LeNet-5, as shown above, was the first neural network to incorporate the idea of a convolution; the idea, as depicted below, that you could connect a neuron in the current layer to only a small, local, subset of neurons in the previous layer.

<img src="/assets/SqueezeNet_diag4.png" alt="fig_4" width="800"/>
(Images courtesy cs231n at Stanford [4])

#### (2) *AlexNet (Krizhevsky et al, 2012)*
<img src="/assets/AlexNet.png" alt="AlexNet" width="700"/>

While neural networks (and CNNs) had been around for more than 20 years by the start of this decade, it was only with AlexNet (shown above [5]) in 2012 that the AI revolution actually began. It did much, much better than traditional CV models, by virtue of:
  * More data (ImageNet contained 1.2 million labelled images)
  * The ability to use GPUs to accelerate training
  * The development and/or use of crucial techniques that helped optimize the network better, such as:
      * ReLU (an activation function that mitigates the vanishing gradient problem)
      * Dropout (the idea of randomly setting the output of a neuron to 0 with a defined probability, forcing the network to learn better representations with some redundancy)
      * Data augmentation (Applying transformations like reflections, translations and changing RGB intensities to increase the effective size of the training data)
      * Normalizing responses locally
      * Only having GPUs communicate in some layers 

#### (3) *VGGNet (Simonyan et al, 2014)*
(A part of the VGG16 network, image courtesy of NYU CS[7])
<img src="/assets/vgg_16.png" alt="vgg" width="600">


By today's standards, AlexNet was a pretty shallow network; it only had 8 layers, but had 60 million parameters (mostly concentrated in the fully connected layers towards the end). 

VGGNet[5] was developed with the idea of increasing depth in mind, using *smaller convolutional kernels* (3x3, as opposed to the 11x11 kernels used in the first layer of AlexNet) to keep the model size in check; it was also one of the first architectures to have **multiple** convolutional layers between pooling layers (AlexNet had a pooling layer for every conv layer), as we can see below. Despite the smaller kernels, however, VGGNet continued the trend of *increasingly high* numbers of parameters, with some variants having more than **130 million**! 