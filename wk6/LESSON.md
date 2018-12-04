# Lesson 6: Regularization and CNNs

## Regularization
(from last lesson)

Often in time series data (in academia) you'll have a series of points in time. With that limited amount of data, an RNN makes sense. (think econometrics)

But in practice you have a lot more data than that! You can add a ton of meta data like day of week, day of year... `add_datepart()` adds this for you. This orovides your CNN with all sorts of useful data.

This means you'll likely be able to treat your time series data just like any other tabular problem.

### Pre-processors
Pre-processors are like transformers but run once before you do any training. They run once on the training set and then any kind of state and meta data that's created is shared with the validation set.

(Transformers run on every batch).

Example preprocessors:
`Categorify`: convert categories with numerical values

`FillMissing`: create a column for NaN values

`Normalize`: normalize continuous variables

> Think carefully about which should be categorical variables. Some numbered variables may be best understood as categorical and not continuous

> When predicting sales, population etc. you often end up using `log=True` while measuring with RMSE.

Don't worry about too many parameters. Use *regularization* to prevent over-fitting.

> `Dropout: A Simple Way to Prevent Neural Networks from Overfitting`

Dropout. We randomly throw away some of our activations. For each minibatch, we throw away a different subset of activations in our hidden layers. How many? with a probability `p` used per minibatch.

This makes sure that no activation can 'memorize' your dataset.

```py
ps=[0.001, 0.01] # provides dropout
emb_drop=0.04
```

Why special dropout in embedding layer `emb_drop`?

Our embeddings are just another layer (matrix multiply by one-hot encoded matrix). So we just remove some of the embedding activations.

### Batch Norm
Part training helper, part regularization...

Speeds up training a lot!

> `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`

We didn't know why BatchNorm worked well until 3 years after.

Essentially, BatchNorm makes our loss landscape less bumpy. This means we can use a higher learning rate :-)

Algo:
- take a minibatch
- take a mean of activations
- find variance of activations
- normalize
- add a vector of biases and multiply by biases
- (both bias layers are learned params via SGD like any others)

Why does this work???

The value of our predictions is f(w1, w2 ... w100000, X)

Our neural network is really just like a big function. What if we added some bias and multiplied by some bias? This lets our model shift the outputs up and down and change the slope. This is all Batch Norm is.

> Prefer weight decay to L2 Regularization

### Data Augmentation
One of the least studied forms of regularization.

Often you can do data augmentation and it wont cause training to take longer or cause overfitting.

> How can we do data augmentation in other domains? Like text data...

## Convolutions: Creating Heatmaps to Understand how Our CNN Decided to Classify an Image

We are going to replace our normal weight matrix (matrix multiplies) with convolutions.

Convolutions are a just a special type of matrix multiply.

Convolution does an element-wise multiplication of a set of pixel values by our `kernel` and adds all the values together.

This gives us a single value for each pixel group.

> https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c

Kernels in convolutions tend to match the data being input.... For example, if we have a 3 channel input, we'll have kernels of 3 dimensions that we convolve over our 3 channel input. This gives us an output the size of our channels (let's say, 5).

We can pass in our input through as many convolutional kernels as we want. Our output is height x width x num_of_kernels_we_chose

Let's say we have a 5x5x16 output. This is a 16 channel output.

In order to control the memory footprint, we add a `stride` to our convolution. For example, we don't do every pixel in our image, but every other pixel (a stride 2 convolution).

When we do a stride 2 convolution, we create twice as many kernels. This doubles the channels of our output.

> Hooks: Pytorch functionality that lets us hook into a given moment in the model and run arbitrary Python.

## GAN / Ethics and Data Science
New sounds, new videos, new texts. Deep Learning crushes this area, esp last 12 months.

We can generate realistic-looking videos and images/audio/text.

> Artificial Intelligence Needs All of Us (Rachel's TED Talk)

Beware! Datasets are heavily skewed to their parent country: US/GB (now a bit China).

> Imagine a mass surveillance system where large quantities of people are much more likely to be misidentified!

> Algorithms are being used for public policy, but these things are not necessarily good. Sometimes the data is based on issues that are out of your control.

> Book: Weapons of Math Destruction

It's really hard to get a non-technical audience to understand the limits of an algorithm so they can develop a smarter decision making process.

People give a lot of credence to the algo's output. Often algos are operated without an appeals process.

Consider how your tech could be used:
- for trolling/harassment
- authoritarian governments for surveillance
- for propaganda or disinformation

> Evan Estola of meetup.com, When Recommendation Systems Go Bad – MLconf SEA 2016. Be careful with your runaway feedback loops!

Only 0.3-0.5% of the world knows how to code.

If you can also do Deep Learning, that's an even smaller slice.
