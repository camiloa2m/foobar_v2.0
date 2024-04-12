# deepBaR: FooBaR attack implementation on different CNN 

Original [FooBaR attack](https://arxiv.org/abs/2109.11249) implementation can be found at:
https://github.com/martin-ochoa/foobar

Some of the code is based on:
https://github.com/DominikBucko/foobar_v2

## Brief description v.1 (ResNet18, VGG13-BN, MobileNetV2)
The purpose of the experiments is to methodically target one layer at a time after the ReLU application. To avoid attacks on all target images, we set a failure probability for the class target to be attacked, and we randomly attack an entire channel for the attacked layer. This attacked channel remains fixed during training, specifically for the case of attacking ReLUs in convolutional layers. For ReLUs in linear layers, we use the same approximation as the original FooBaR attack. In the end, we obtained a trained model for each attack. Finally, we generated some fooling images using gradient descent and obtained metrics for the attacks.

## Brief description v.2 (VGG19, ResNet50, DenseNet-121)
... to continue! :v


### >>> Repository under construction!

