#! /bin/bash

gcloud compute scp --recurse ./data fastai-instance:~/data
