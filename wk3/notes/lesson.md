# Lesson 3

### Fast Iteration
Use small image sizes. For example 128 instead of 256.

You could then use transfer learning from your first model (trained on the original smaller dataset). Transform your dataset to the large original size. This is basically a NEW DATASET as seen by the learner.

So we can create a new databunch with 256 images.

```py
learn.data = new_databunch
learn.freeze() # so we go back to training only the last layers
learn.lr_find()
```

> Progressive resizing: we could deliberately make a lot of smaller datasets to step up from in tuning, say 64x64, 128x128, 256x256, 512x512 etc…

### Learning Rate Finders
When unfrozen, slice from 10x back before the bottom (min) and second value of frozen lr rate / 5

### Kaggle
Awesome datasets you can use.

### Fine-tuning a production model
Imagine we have some errors from a production model

- Re-load in your model
- Create a databunch from your mis-classified data instances
- Train on them

### Multilabel Classification
Metrics may need to be different for this model, because we're not just doing a binary classification. e.g. `accuracy_thresh` and `fbeta`


> partial (python3)
> new_func = partial(function, keyword_arg=keyword_value)
> # creates a new, customized version of the function that always uses the specified keyword arg

### DataBlock API
> chain methods to process your data

DataSet: from pytorch. Abstract base class to have your data inherit from.

Starting point for your data is a DATASET. We first just need to be able to say:
- what is the nth element in my dataset?
- how many items are in this dataset?

We can't train a model with dataset alone.

Minibatch: a few items that we present to the model at a time so we can train in parallel on the GPU.

To create a minibatch, we use a Pytorch class called a DataLoader.

DataLoader --> converts a dataset into minibatches for your GPU!
- takes a DataSet
- grabs items at random
- makes a batch of the size you specify
- serves it up to the GPU

But even with our DataLoader, we don't have enough to train a model!

DataBunch binds together a training DataLoader and a validation DataLoader!
- gives us our training and validation sets

Example:
```py
ImageFileList.from_folder(path)
  .label_from_folder()
  .split_by_folder()
  .add_test_folder()
  .datasets()
  .transform(tfms, size=224)
  .databunch()
```

### Camvid
Start with raw image. End with masked image where all discrete elements are the same color (tagged).

Segmentation datasets are a TON OF WORK, depending on domain because someone has to label all the elements in the image.

use `open_mask` to see the labeled mask of the image

If we transform our X when working with a mask, we need to transform out Y as well (the mask) so they line up correctly!

### UNET
UNET actually looks like a U... or at least the diagram does :)

Originated for biomedical image segmentation.

> Often your loss will go up a bit before going down again...

> Learning Rate Annealing: decreasing your learning rate over time
> Leslie Smith realized that gradually increasing the learning rate at first helps the model explore the fitness landscape.

If your loss increases a little bit before decreasing significantly, you've found a great max learning rate!

For `learn.recorder.plot_losses()`

You want to see a shape that goes up at first and then declines.

> Mixed Precision Training:
If you're running out of data a lot, you can convert from double precision numbers to single precision `.to_fp16()`!

Funny enough, .to_fp16(), actually improved the performance. Why?

> Sometimes by making things less precise in deep learning the model actually generalized BETTER.

> There should be a card set of these little tricks.. to spur creativity as you go.

### Head Pose
Tensor with X,Y coords for center of face. Given to us by the data set preparer.

Determines how the model measures its success.

Since we're not classifying (we're doing regression) we'll use a different loss function.
learn.loss_func = MSELossFlat()

This is *IMAGE REGRESSION*, predicting a continuous value from images.

### IMBD (NLP)
Can we create a classification of documents?

IMBD has tons of movie reviews. Classified as negative or positive.

After we create a DataBunch (using factory or datablock api) the model *TOKENIZES*.

> Tokenization: Normalizes contractions, punctuation etc. Makes sure that each TOKEN represents a single linguistic concept.

> Numericalization: converts all words to numbers. xxunk represents token model doesnt understand

NLP classifier is similar to creating any other model, really.

#### Linear Classifiers
No matter how many matrix multiplications we do, we'll never get to something incredible. A linear function on top of another linear function is just another linear function.

No amount of stacking matrix multiplications is going to solve our problem!

> Non-linearities: we take the result of a matrix multiplication and pass it through an ACTIVATION. This is a non-linearity that transforms our linear functions into something non-linear.

Different activations have different properties. In the past we used a sigmoid. Today we often use Rectified Linear Unit (ReLU)... `max(x, 0)`. We replace negatives with 0.

This transformation creates a DL model.

*Universal Approximation Theorem*! If we have stacks of linear functions and non-linearities, we can approximate any function arbitrarily closely. With a big enough matrix or enough matrices, we can approximate any function.

input -> weight matrix -> replace negatives with zeros -> weight matrix -> replace negatives with zeros ... on and on, then we update our weights using SGD! Do this enough and we end up with a generalizable model.

This is essentially it!

Images get optimized using convolution.
Text/language models get optimized using RNN (recurrent).
