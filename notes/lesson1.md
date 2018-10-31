# Set up
GCP $300 free credit

Conv neural nets: https://cs231n.github.io/convolutional-networks

> doc()
> get docs for a function

## First Stage: Image Recog. Training
DataBunch (general class for handling data, passed into learners)
- splits into training/validation set
- makes sure you don't mess with the validation set during training

resent34 vs resnet50, you often choose between these architectures

`ConvLearner`

`metrics` things printed out during training

> Transfer Learning:
Taking a pre-trained model that does something pretty well and adapting it to your use-case so it does that, really, really well.

> Validation Set:
> A set of images that your model does not get to look at

```py
learn = ConvLearner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)
# One cycle learning is some new innovation that works better than typical .fit training
# 4 (cycles) represents the number of times that we go through the full data set

learn.save('my-filename')
```

After training, we can check out how the model performed with `ClassificationInterpretation.from_learner()`

> High Loss
> When was wrong about something and very confident about it

```py
interp.most_confused(min_val=)
# Give us the categories that the model is most confused about
```

The ClassificationInterpretation let's us sanity check the model results.

## Second Stage: Unfreezing Fine Tuning and Learning Rates
To quickly train a model, we often freeze most the layers and just train the final rows. This let's us train fast.

```py
learn.unfreeze()

learn.fit_one_cycle(1)
```

Sometimes unfreezing this will make the model worse!

> resnet34 has 34 layers
> resnet50 has 50 layers

Each layer combines the previous layers, adding complexity and specificity to what's output.

Since early layers are so general/abstract, when we unfreeze everything we get worse performance. We don't want to update the EARLY layers, which are general/abstract, the same way we update the final layers which are highly specific.

```py
learn.lr_find()
# How fast can we train this model without destroying it?
```

The learning rate finder helps us find the optimal learning rate (fastest where we minimize error).

```py
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-4))
# Trains early layers at 1e-6 and later layers at 1e-4
# The max_lr will be distributed between the layer
```

### Creating Labels

ImageNet-style data organization often puts the files in a folder with their label as the folder name

```py
ImageDataBunch.from_folder(path)
ImageDataBunch.from_csv(path)

# Check notebook for regular expression code

# Arbitrary label functions also supported for extraction

# From an arbitrary array
ImageDataBunch.from_lists(path, fn_paths, labels=labels)
```

PLAN FOR STUDY:
1. Blog post with little project to illustrate ideas per week
2. Post on Medium/website/Kaggle
3. One big project at end with launch

IDEAS:
type in alt text in natural language, get a source photo. semantic distance from existing unsplash tags

What do users who eventually convert look like

What about understanding that a user is having trouble and then proactively getting them a support person?

Predicting classical genres (baroque, classical etc. from music spectrograms)

Predicting movie sales from trailers
Predicting game sales from trailers

Predicting exit intent or abandoned cart behavior from user path data (preemptively get a salesperson to help)
Predicting 'big fish', what high value converted users look like when they use your site

#### Code for the Notebook
```py
%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai import *
from fastai.vision import *

help(untar_data)

path = untar_data(URLs.PETS)
path_anno = path/'annotations'
path_img = path/'images'

fnames = get_image_files(path_img)
print(fnames[:5])

pat = r'/([^/]+)_\d+.jpg$'
# Regex for extracting labels from file paths

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms, size=224)
data.normalize(imagenet_stats)
# Normalize brightness for images so model not misled by intensity of a given image
# imagenet_stats are the constants we use to normalize

data.show_batch(rows=3, figure=(7,6))

learn = ConvLearner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('my-filename-here')

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(0, figsize=(15,11))
interp.most_confused(min_val=2)

learn.unfreeze()
# unfreeze the layers that were pre-trained in resnet
# often the model will get worse when unfrozen because we are training all layers at once

learn.lr_find()
# find the learning rate

learn.recorder.plot()
# all learners have a recorder embedded in them, which we can use to check out the loss rates at different learning rates

```

What to do when the model is confused:
- set weight for unbalanced classes in loss function
- data augmentation
- re-balance classes by training again with more of the problematic ones
- freezing all layers except for last ones


Flashcards:
```
help(func_name) --> get docs about a function

learn = ConvLearner(data, models.resnet34, metrics=error_rate) ==> create a new convolutional learner
learn.save(my_filename) ==> save my trained model

untar_data()

ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224) --> Create an DataBunch of image data using a regular expression to extract labels from the file name
1. ds_tfms=get_transforms() ==> param in ImageDataBunch that zooms, centers, and crops images. applies these transforms automatically to our images
2. size=224 ==> size that we want images to be. Since resnet34 was trained on images of size 224x224, we want our images to be the same size

```


1. Import deps
2. import data source
3. get labels from data/somewhere else
4. create an ImageDataBunch and pre-process the images a bit
5. Create a ConvLearner with model, and other params
6. Train the learner for some number of epochs
7. Create Interpretation from learner
8. Look for most confused values to see what's going wrong in the model
9. FINE TUNING!
