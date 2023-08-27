# FooBaR_v2.0: FooBaR attack implementation on different CNN (ResNet18, VGG13, MobileNetV2)

Original [FooBaR attack](https://arxiv.org/abs/2109.11249) implementation can be found at:
https://github.com/martin-ochoa/foobar

The code for training and generating the fooling images is based on:
https://github.com/DominikBucko/foobar_v2

## Brief description
The idea of the experiments is to systematically attack one layer after the ReLU application, one layer at a time. We set a failure probability for the class target to be attacked in order to avoid attacks on all target images, and we randomly attacked an entire channel for the attacked layer, this attacked channel being fixed during the training. The latter for the case of attack ReLUs in convolutional layers. In the case of ReLUS in linear layers, we use the same approximation of the original FooBaR attack. In the end, we obtained a trained model for each attack, on which we then generated the fooling images using gradient descent and obtained some metrics of the attacks.


### >>> Repository under construction!

