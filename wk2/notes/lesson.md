# Lesson 2
## Computer Vision – Deeper Applications

> Watch videos 3 times

What we'll learn:
  Computer Vision to
    NLP to
     Tabular to
      Collaborative Filtering to
       Embeddings to
        Comp Vision Deep Dive to
         NLP Deep Dive

# Lesson 2 Download Notebook

Random seed whenever we do a random number of elements into an ImageBunch.

## Learning Rate
For learning rate, we want the strongest downward slope that's sticking around for a while!

### Going to Production
Use a CPU not a GPU. CPUs are good at doing a lot of things at the same time.

Most apps that are not at some enormous scale, we use CPU not GPU.

Get folders from classes, one single data at a time.

We are doing inference, one photo at a time --> to determine which class it is.

> This week, take a model to production!

### When deploying
> Your training loss should never be higher than your validation loss.

Few things that go wrong:
- too high of learning rate (valid loss will explode)
- too low of learning rate (training loss will be higher than validation loss)
- too few epoch (training loss will be higher than validation loss)
- too many epochs (over fitting. your error will start to grow)

As you overfit, your error will get better and better until it starts getting worse.

> error rate: 1 - accuracy

> accuracy is

# MATH!
Predictors are functions of pixel values. Think of a huge matrix of pixel values.

The matrix is going to return probabilities from the possible values. `np.argmax` or `torch.argmax` is telling us to to find the class with the highest prediction value.

```
y = ax + b
y = a1x1 + a2x2 ; x2 = 1
We imagine we are multiplying by 1.

dot product = summing up a bunch of things multiplied together
matrix product =
ybar = Xabar
```

matrixmultiplication.xyz

### Unbalanced Classes
What do I do with unbalanced data? Just do it! It's probably OK :-)

Or just make a few copies of the unbalanced class. This is called over sampling!

## SGD
> Tensor: it basically means an array of a regular shape :-)
> tensor RANK means how many axis there are.

An image is a rank 3 tensor.
A vector is a rank 1 tensor.

Stochastic Gradient Descent!

Computer will try things until it fits our data

An image is a 3-dimensional tensor.

SGD is how we get from these insane matrices of pixel values to an accurate prediction.

Regression problem is predicting a continuous variable.

> FLASHCARDS from SGD notebook vocabulary section

Our linear regression example is essentially the same as SGD in DL, except DL uses mini-batches. We calculate loss using batches and keep updating weights.

Each epoch refers to a full pass through all the mini-batches (all images). The training loop is going through mini-batches one at a time and updating weights.

### Regularization
All the techniques that we use when training our model to make sure the model works well on the data it has seen AND the data it hasn't seen yet.

Always check models on a validation set.
