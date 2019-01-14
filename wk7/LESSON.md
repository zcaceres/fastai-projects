# Lesson 7

## Resnet MNIST CNN
Starting out, we can get to high accuracy on MNIST with a simple architecture. Basically, a couple stride 2 convs with Batchnorm and ReLUs.

How can we make it better? A *deeper* network.

Interestingly in `Deep Residual Learning for Image Recognition`, we found that a deeper network of 56 layers performed worse than a 20 layer network. This makes no sense (at least on the training set) because that network should start to seriously overfit due to overparameterization.

Researchers that found this proposed a Residual Learning building block (RES block).

Basically, we add the input to the output.

This is the `identity connection` or `skip connection`:

```
out = conv2(conv1(x)) # original
out = x + conv2(conv1(x)) # with residual
```

In theory, this network has to train at least as well as the 20 layer network because it effectively contains the network. The convs could have weights set to zero for all but the first 20 layers and the input could be passed through the last 36 layers. There is a path to skip all the additional convolutions. This idea crushed imagenet.

3 years later a paper was written to understand WHY this works.

### DenseNet
Dense blocks varied this idea.
DenseNet: basically identical to ResNet but the + is replaced with concatenate when we do the residual.

A dense net will get bigger and bigger and bigger as inputs are concatenated through the layers.

#### Segmentation with UNet
We start with a Resnet34 (pretrained model).

UNet downsamples through convolution. We end up with a much smaller grid than we started with. How do we do computation that increases the grid size so we arrive back at our original grid size?

We use a `stride half` convolution. Also known as a `deconvolution` also known as `transposed convolution`. This is the same as a convolution but with *padding added between the elements in our input grid`.

ex: if we put in a 2x2 grid and did a deconvolution with padding of 1, we would get a 4x4 grid as output!

A faster more modern way to do this is just to double the existing pixel values to make a larger grid.

`Nearest-neighbor interpolation`
```
a b           a a b b
c d   ====>   a a b b
              c c d d
              c c d d
```

Then we can just do a stride 1 convolution which preserves our larger grid size.

Originally, models just did this. It works OK but not great, because once the data is downsampled, it's insane to try to reconstruct the data at its original dimensions. Too much as lost.

Enter UNET.

Unet introduced skip connections between the downsampling path to its matching step in the upsampling path. They used a concatenated skip connection, like a dense block.

#### Super-resolution with UNet
- crappify images by adding a watermark and lowering overall quality
- trained with MSELoss
- output removes watermark pretty well but resolution is still not great. Why?

Well, the difference between the output pixels and the ground truth pixels is not that much. More or less the pixels are right. But not right enough to capture the texture and edges that we need to see the photo as high-resolution.

#### GANs
Typically we use GANs to solve this problem.

> A GAN's loss function calls another model.

The other model is called the `discriminator` or `critic`. This model classifies the output – this is the low res image, or this is the high res image.

The loss function becomes: `how good is our model at fooling the critic`?

Once our generator gets pretty good, we train the CRITIC some more. As it gets better, we go back and train the generator. Back and forth, each side getting better.

Typically, it's considered hard to deal with GANs. Without pretrained models, the generator and critic both suck at the start. It's the blind leading the blind.

`GANLearner` takes in a generator and a critic.

#### SuperRes: Can we get rid of GANs?

`feature losses`:
What if after we ran our image through the encoder/decoder we put it through a pretrained imagenet network. Then we take activations from somewhere in the middle of the network.

We also pass the target through the pretrained imagenet network and grab activations from the same spot. We do an MSE comparison of our encoder/decoder at that activation point and our target's activations at that point.

#### RNNs
An RNN is basically a refactoring of a traditional fully connected model.

Instead of defining every layer, we use a loop to pass our data through the same layer some number of times.

We can use this to predict any subsequent element in a sequence by grabbing the prediction output after each pass through the loop.

If we predict only the final word, we'll get better performance than predicting every word because we have more state to base our prediction on.

For a real RNN we save the state of our last iteration and use it to predict the next iteration.
