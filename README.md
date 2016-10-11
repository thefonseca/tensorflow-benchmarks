# TensorFlow Benchmarks
Benchmarks for Deep Learning [models implemented in TensorFlow](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models). These benchmarks are easy to reproduce if you already have TensorFlow installed on your machine. If you have a different hardware, feel free to contribute.

For more in-depth benchmarks, see:
* [convnet-benchmarks](https://github.com/soumith/convnet-benchmarks)
* [DeepBench](https://github.com/baidu-research/DeepBench)

----

### AlexNet

Timing [benchmark for AlexNet](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/alexnet/alexnet_benchmark.py) inference. For more details refer to the paper [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

Running the benchmark:

```
python tensorflow/models/image/alexnet/alexnet_benchmark.py
```

| Num | GPU/CPU               | Memory              | Forward pass (ms) | Forward-backward pass (ms) | Details |
|-----|-----------------------|---------------------|-------------------|----------------------------|---------|
| 1   | Titan X               | 12GB GDDR5          | 70 +/- 0.1        | 244 +/- 30                 | as reported in [alexnet_benchmark.py](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/alexnet/alexnet_benchmark.py) |
| 2   | GeForce GTX 960M      | 2GB                 | 121 +/- 0         | 359 +/- 1                  | Dell XPS 15 9550 / Ubuntu 16.04 / CUDA v7.5 / cuDNN 5.1 |
| 3   | K40c                  | 12GB GDDR5          | 145 +/- 1.5       | 480 +/- 48                 | as reported in [alexnet_benchmark.py](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/alexnet/alexnet_benchmark.py) |
| 4   | GeForce GT 750M       | 2GB                 | 536 +/- 2         | 1466 +/- 18                | MacBook Pro Late 2013 / OS X 10.11.6 / CUDA v7.5 / cuDNN 5.1 |
| 5   | 2.3 GHz Intel Core i7 | 16 GB 1600 MHz DDR3 | 2473 +/- 34       | 7091 +/- 117               | MacBook Pro Late 2013 / OS X 10.11.6 / CUDA v7.5 / cuDNN 5.1 |


----

### CIFAR-10
Benchmark for image recognition using a convolutional neural network (CNN) in CIFAR-10 dataset. For more details refer to the [TensorFlow tutorial](http://tensorflow.org/tutorials/deep_cnn).

Running the benchmark:

```
python tensorflow/models/image/cifar10/cifar10_train.py
```

| Num | GPU/CPU               | Memory              | Examples/second | Seconds/batch    | Details |
|-----|-----------------------|---------------------|-----------------|------------------|---------|
| 1   | GTX 1080              | 8 GB GDDR5X         | 1780.0          | 0.072            | as reported in [tf benchmarks](https://github.com/tobigithub/tensorflow-deep-learning/wiki/tf-benchmarks) |
| 2   | GeForce GTX 960M      | 2GB                 | 1529 +/- 68     | 0.0839 +/- 0.0041| Dell XPS 15 9550 / Ubuntu 16.04 / CUDA v7.5 / cuDNN 5.1 |
| 3   | Titan X               | 12GB GDDR5          | 550.1           | 0.233            | as reported in [tf benchmarks](https://github.com/tobigithub/tensorflow-deep-learning/wiki/tf-benchmarks) |
| 4   | GeForce GT 750M       | 2GB                 | 535.33 +/- 15   | 0.2393 +/- 0.007 | MacBook Pro Late 2013 / OS X 10.11.6 / CUDA v7.5 / cuDNN 5.1 |
| 5   | 2.3 GHz Intel Core i7 | 16 GB 1600 MHz DDR3 | 336.6 +/- 15    | 0.38 +/- 0.015   | MacBook Pro Late 2013 / OS X 10.11.6 / CUDA v7.5 / cuDNN 5.1 |
