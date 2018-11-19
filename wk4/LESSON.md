# Lesson 4

### NLP
The hard thing about using a neural net for NLP is getting to the point at which your model "speaks English". The training of the raw network is extremely difficult before you can do anything interesting (like sentiment analysis).

Self-supervised learning: no external labels but the labels are in the data itself (such as guessing the next word in a sentence).

NOTE:
```
RGB --> weight matrix layer --> vector --> non linearity (RELU) --> vector ---> on and on!!


Our first layer may have random number weights.

Whether a nonlinearity or a weight matrix, we have a LAYER. Numbers that are the result of our calculations are activations. Numbers that are used to calculate are WEIGHTS, parameters.

We can keep doing this until we reach the point that we're ready for our output, a vector determined by the needs of our model. For example, digit recognition would be a vector of 10 possible digits.

Gradient descent changes the weights in order to optimize our loss.

Our loss takes our final layer of activations and compares them to our actuals.

<!-- Given our predictions and our target (truth) we take the derivative of -->
```

### Softmax
Instead of stopping after the last activation, we might add one more activation function: Softmax. Sum of all our targets is going to be 1 when we're doing single label classification.

Max will be 1.
Min will be 0.

For every output, we exponentiate each value. Then we divide them by the sum to get them to add up to 1.

> Softmax shouldn't be used when you want it to be possible for the model to say "I'm not sure".

### Cross entropy loss
actual * log(prediction) for every category

Any loss function will use the predictions and the actuals.

Punishes the model based not just on its predictions but how certain it was of a given prediction.


## NLP
Old versions of NLP used NGRAMS (pairs of words).

### Movie Reviews
A whole bunch of words. At the end, we determine whether positive or negative sentiment.

The neural net that does this is called a `Classifier`. There's a slight difference in these networks in that the reviews can be of different sizes whereas images are a standard size.

> Still just layers of matrix multiplication with RELUs !

Until about 12 months ago, neural nets did not do this problem well. This is because the language data sets are too little information to get to a reliable outcome. So we need some sort of pre-trained model. A LANGUAGE MODEL, which learns how to predict the next word in a sentence.

How Jeremy solved this:
1. Wikitext 103 Language Model. Predicting the next word of English from Wikipedia articles
2. Fine tuning from IMDB. Predicting the next word in a movie review...
3. Discard the last few layers and replace them with some random layers.
4. IMBb classifier has two activations that come from their final layer --> probability of positive and probability of negative

To get to our sentiment model, we...
1. tokenize the text
2. generate a vocab (a big list of every unique token)
3. Throw away the ones that don't appear enough. Basically, get the top 60k words.
4. Numericalization: convert our vocabulary to an integer that represents each word

> Rules of thumb for training RNNs are different than CNN:
> momentums(0.8, 0.7)

> You can use Random Forests to find optimal hyperparameters.

### Tabular Data
Using neural networks for tabular data was considered an absurd idea up until 12 months ago. gradient boosting, random forests, and regression were the recommended approaches.

Turns out that Neural Nets are great at tabular data.

> feature engineering is much simpler with neural nets, rather than feature engineering for other models. For example, Instagram used a neural net for their home page and it was more maintainable a more accurate.

> Jeremy used to use random forests 99% of the time for tabular data. Now he uses NN 90% of the time. For the 10% of cases where he wouldn't use them... first give them a try and see how it goes. Might as well try both and see how it goes. Depending on which is better, stick with it.

```py
dep_var = '>=50k'
cat_names = [variables that are a selection of a set of possibilities]
cont_names = [numbers, continuous vars]
procs = [FillMissing, Categorify, Normalize]
# preprocessors. Similar to Transforms
# FillMissing deals with missing values. Categorify turns categories into Pandas categories. Normalize will normalize our continuous vars. For missing data we replace with median and add a new column that shows they're missing. Make sure you do this for both test and validation so they match.
```

If data has some sort of structure (time series etc.) we should grab slices next to one another for training and validation.

### Collaborative Filtering
When you have information about who bought what or who liked what. Most basic version, UserId: MovieId --> this user bought this movie.

You can add info, user gave a review at this time.

> NOTE: Collaborative filtering is best when you already have some data to use about a given row. Otherwise you get the 'cold start problem': not being able to predict before you have any data.

Models being created by fastai is a Pytorch `nn.module`. Always look for the .forward() method. This is how the model is run. Pytorch calculates our gradients for us.

Pytorch gives us a lot of standard NN setups. Embedding is one of those.

#### Embeddings
An embedding is a matrix of weights. Specifically, a matrix of weights which you can look into and grab one item out of. It's a weight matrix that is designed for your to index into and grab a single weight out of it.

For our movie/users example, we have an embedding matrix for users and for movies.

> Bias terms: in practice, we add a bias to our matrix multiplication. We do this to make it easier for our network to learn the right thing.

Flashcards:
- Inputs
- Weights/parameters
- activations
- activation functions / nonlinearities
- output
- loss
- metric
- cross-entropy
- Softmax
- fine-tuning
  - layer deletion and randm weights
  - freezing and unfreezing

> very common to have an activation function at the end of your network

## Leslie Smith Talk
What if instead of getting an exact optimal rate we got upper and lower bounds instead? This is how we got cyclical learning rates.

learning rate finder (range test), quite quickly you can figure out an optimal learning rate.

Got involved to DL in 2013.

#### One-Cycle Learning
If you gradually increase the learning rate and then decrease it for a while you can get to higher learning rates. And at the same time you decrease the momentum and then increase the momentum.

At start and end: low learning and high momentum
In the middle: high learning and low momentum

`learning rate * weight decay (TBS * 1 - a))~~constant`

#### Weight Decay
one of the oldest regularization techniques.

> I start with a very large weight decay. I decay separate from learning rate. I end the training with 0 weight decay!

Why would one form of regularization (weight decay) you want to decrease and another form (drop out) you want to increase! This is an effective but strange finding.

#### The Problem of Incremental Learning
Base case we have a lot of training data. In production we get classes that we didn't train on. We need our model to recognize that we don't know this class.

Few-shot learning: we use a second, pre-trained network to recognize the extra classes.

#### Train with Small Pictures and then Make Them Bigger
This seems really dumb and obvious. Most models cant do this because their max pool is a fixed size at the end of the model.

Train really fast at first. It gets slower as you increase the size. And you give the model 'new data' each time you train since the image has changed.

By the end, the weights reflect a full-resolution photo! Which is what our predictions would be feeding in anyway.


#### Dynamic Data Augmentation
Networks are blank slates. They have to learn everything. Contrast, brightness.. they have to know that this doesn't matter to identify it.

> Take a small subset of the training set and use aggressive data augmentation on it.

What's the FASTEST way to train a network so that it's good for transfer learning?

#### Curriculum Learning
a particular kind of dynamic learning where you start with an easy question and gradually make it harder.

> i.e. less drop out, a subset...


> ADAM W... L2 regularization

#### Transfer Learning with Tabular Data
You can reuse categorical embeddings elsewhere in the business. You can re-use these embeddings when making a model elsewhere. Pre-train first layer of embeddings and then training from there.

> There's not that much of a big difference in research and business. A lot of time is spent trying to predict/optimize things.

#### Priya Goyal
- take last batch norm, which has two parameters (amount you multiplied by and amount you subtract)
- all of the tricks that let you use large batch sizes also let you use large learning rates
  - kfac
  - lars
  - batch norm 0
  - resnet/resblock scaling
  - NMinnorm Training
