# Deploying a Deep Learning Image Classification Model with NodeJS,  Python, and Fastai

> TL|DR: Use [this](https://github.com/navjotts/node-python/tree/example_ml_image_classification
) to easily deploy a FastAI Python model using NodeJS.

You've processed your data and trained your model and now it's time to move it to the cloud.

If you've used a Python-based framework like [fastai](https://github.com/fastai/fastai) to build your model, there are several excellent solutions for deployment like [Django](https://www.djangoproject.com/) or [Starlette](https://github.com/encode/starlette). But many web devs prefer to work in NodeJS, especially if your model is only part of a broader application.

My friend [Navjot](https://github.com/navjotts) pointed out that NodeJS and Python could run together on the same server if we could send [remote procedure calls](https://en.wikipedia.org/wiki/Remote_procedure_call) from NodeJS to Python.

I extended his shared NodeJS/Python environment into [a simple, minimal boilerplate](https://github.com/navjotts/node-python/tree/example_ml_image_classification) for a NodeJS deployment of an image classification model. The deep learning model was made with the [fastai](https://github.com/fastai/fastai) library. Although fastai and our model were built in Python, we can expose the model to users from NodeJS.

Here's how.

## How does it work?
The deployment works using three main modules: `server.js`, `PythonConnector.js`, and `PythonServer.py`.

Our Express server in `server.js` provides a standard, RESTful API to the outside world. Nothing new here!

But our Express server also references our `PythonConnector`, which serves as a middle man between the worlds of NodeJS and Python.

```js
const PythonConnector = require('./PythonConnector.js');
```

Specifically, on startup `PythonConnector` spawns a python3 process that sets up our `PythonServer`. It also negotiates and maintains a socket connection to `PythonServer` through [zerorpc](https://www.zerorpc.io/).

```py
class PythonConnector {
    static server() {
        if (!PythonConnector.connected) {
            console.log('PythonConnector – making a new connection to the python layer');
            PythonConnector.zerorpcProcess = spawn('python3', ['-u', path.join(__dirname, 'PythonServer.py')]);
            PythonConnector.zerorpcProcess.stdout.on('data', function(data) {
                console.info('python:', data.toString());
            });
            PythonConnector.zerorpcProcess.stderr.on('data', function(data) {
                console.error('python:', data.toString());
            });
            PythonConnector.zerorpc = new zerorpc.Client({'timeout': TIMEOUT, 'heartbeatInterval': TIMEOUT*1000});
            PythonConnector.zerorpc.connect('tcp://' + IP + ':' + PORT);
            PythonConnector.connected = true;
        }
        return PythonConnector.zerorpc;
    }
  ...
}
```

When the client makes a request to a given endpoint from our `server.js`, such as `/predict`, our Express server commands the `PythonConnector` middleman to invoke a function in our Python environment via zerorpc.

Our Python environment returns some JSON, which can be processed and forwarded along to our client.

Here's how that looks when a user sends an image for classification.

First our server gets the request...

```js
...

// Our prediction endpoint (Receives an image as req.file)
app.post('/predict', upload.single('img'), async function (req, res) {
    const { path } = req.file
    try {
        const prediction = await PythonConnector.invoke('predict_from_img', path);
        res.json(prediction);
    }
    catch (e) {
        console.log(`error in ${req.url}`, e);
        res.sendStatus(404);
    }

    // delete the uploaded file (regardless whether prediction successful or not)
    fs.unlink(path, (err) => {
        if (err) console.error(err)
        console.log('Cleaned up', path)
    })
})

...
```

... and calls out to `PythonConnector`...

```js
static async invoke(method, ...args) {
    try {
        const zerorpc = PythonConnector.server();
        return await Utils.promisify(zerorpc.invoke, zerorpc, method, ...args);
    }
    catch (e) {
        return Promise.reject(e)
    }
}
```

... which makes a prediction from our fastai model:

```py
from model_fastai import FastaiImageClassifier

class PythonServer(object):
    def listen(self):
        print(f'Python Server started listening on {PORT} ...')

    def predict_from_img(self, img_path):
        model = FastaiImageClassifier()
        return model.predict(img_path)
```

You can use the boilerplate out of the box. The included model classifies black bears, teddy bears, and grizzly bears.

If you'd like to use the boilerplate for your own project, you can customize the model with instructions available on [Github](https://github.com/navjotts/node-python/tree/example_ml_image_classification).

Thanks for reading!
