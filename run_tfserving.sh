#!/bin/bash

/usr/local/bin/tensorflow_model_server --port=9000 --model_name=imagenet --enable_batching --model_base_path=/tmp/model

