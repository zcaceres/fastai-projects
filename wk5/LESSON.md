# Foundations of Neural Networks
How do these things actually work?

## Review
Two types of layers. Layers that contain `parameters` and layers that contain `activations`.

Input activations --> layer activations --> weight matrix

> Is it a parameter of is it an activation?

> Activation functions: an element-wise function. Basically, it transforms every element in the vector by some function and gives you a new vector of the same size transformed by the function.

Output of final layers --> loss func --> output

> Back propagation: weights -= weights * (grad * learning_rate)

### Fine Tuning
In ImageNet we have a final layer of 1000 categories. Your model may not have 1k categories. When we create a cnn with fastai, we automatically discard this final layer and replace two new layers with a RELU in between. Why? Because this last layer is useless to us.

We can infer from the DataBunch how many activations we need to have in this last layer (ex. number of classes in our set).

These new weight matrices are full of random numbers. So we need to train them. We don't do this for all our intermediate layers because these layers learn general features like 'fur' or 'curves' or 'wheels'. These might be useful since they are more general than our highly specific final layer(s).

So we `freeze` our intermediate layers. We only train these final layers then.

Once we have a decent result we `unfreeze`. But we still probably want to train the final layers more than the early layers.

*To do this, we give different parts of the model different learning rates.* This is `using discriminative learning rates`.

```py
.fit(1, 1e-3) # every layer gets same learning rate
.fit(1, slice(1e-3)) # final layer gets the specified learning rate, earlier layers get the same rate which is this number (1e-3) / 3
.fit(1, slice(1e-5, 1e-3)) # last layer group gets 1e-3, first layer group gets 1e-5, other layer groups get rates that are equally spread between these two
```
By default, fastai uses three layer groups for a CNN.

### Affine Function
Basically, matrix multiplication (with some caveats which we'll get to later).

### Embeddings
Multiplying by a one-hot-encoded matrix is identical to an array lookup. Creating embeddings is just a computational shortcut. Embedding: we look it up in the array. It's identical to a matrix multiply between weights and a one-hot-encoded matrix.

> Latent Features: hidden things that were there all along. "This movie has John Travolta"

### Bias
In addition to the embeddings, we add on a bias to our weights. This means (in our collab filtering model) a movie has an overall 'people like this movie' and a user has an overall 'this person (dis)likes movies a lot'.

### Collaborative Filtering
Typically termed: `users` and `items`

`y_range` makes our final activation function a sigmoid from (x, y). You want the range to be a little bit below the minimum to a little bit more than the maximum.

`n_factors` the embedding size (width). Specify it through experimentation.

> principal component analysis: find a smaller number of activations that approximate these activations

### Entity Embeddings (entity matrices in tabular data)
Used in Rossman Kaggle competition. Embedding vectors actually discovered something like 'geography'... what!

### Weight Decay
`wd=1e-1`

Weight decay is a type of regularization.

Models with more parameters tend to 'overfit' (in the view of traditional stats).

But we want lots of parameters to make more complex functions with curvy bits to mimic real life! So what if we use lots of parameters but PENALIZE complexity.

Take our loss function and add the sum of the squares of parameters multiplied by `wd`.

wd should usually be 0.1

`a.sub_(lr * a.grad)`

```
wt = w(t-1) - x dL/dw(t-1)
```


```py
for p in model.parameters(): w2 += (p**2).sum()
```

### MNIST SGD
Task: create your own Pytorch linear class

`model.parameters()`

> Cross-Entropy Loss:


### Adam
optim in Pytorch usually assigned to SGD

```py
optim.SGD(model.parameters(), lr) # our usual optim

optim.Adam(model.parameters(), lr) # add in weigh decay...
```

What is ADAM here???

> Momentum: (derivative * 0.1 * last update * 0.9)

Called momentum because you're using the last update. My gradient plus the exponentially weighted moving average of my last step.

`exponentially weighted moving average`

Momentum is usually 0.9

> assignment: add momentum to one of th early note books

RMS prop is similar to momentum --> exponentially weight
