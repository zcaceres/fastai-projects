#!/usr/bin/env bash

# Recursively zips our 'data' directory into a tarball named 'composition.tar.gz', excluding the wav and mid files we made in previous steps.
tar -czvf compositions.tar.gz ./data --exclude=*.wav --exclude=*.mid
