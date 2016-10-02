# TensorFlow Benchmarks
Benchmarks for Deep Learning [models implemented in TensorFlow](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models). These benchmarks are easy to reproduce if you already have TensorFlow installed on your machine. If you have a different hardware, feel free to contribute.

For more in-depth benchmarks, see:
* [convnet-benchmarks](https://github.com/soumith/convnet-benchmarks)
* [tf benchmarks](https://github.com/tobigithub/tensorflow-deep-learning/wiki/tf-benchmarks)
* [DeepBench](https://github.com/baidu-research/DeepBench)

----

### AlexNet

Timing [benchmark for AlexNet](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/alexnet/alexnet_benchmark.py) inference. For more details refer to the paper [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

Running the benchmark:

```
$python tensorflow/models/image/alexnet/alexnet_benchmark.py
```

| Num | GPU/CPU               | Memory              | Forward pass (ms) | Forward-backward pass (ms) | Details |
|-----|-----------------------|---------------------|-------------------|----------------------------|---------|
| 1   | Titan X               | 12GB GDDR5          | 70 +/- 0.1        | 244 +/- 30                 | as reported in [alexnet_benchmark.py](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/alexnet/alexnet_benchmark.py) |
| 2   | K40c                  | 12GB GDDR5          | 145 +/- 1.5       | 480 +/- 48                 | as reported in [alexnet_benchmark.py](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/models/image/alexnet/alexnet_benchmark.py) |
| 3   | GeForce GT 750M       | 2GB                 | 536 +/- 2         | 1466 +/- 18                | MacBook Pro Late 2013 / CUDA v7.5 / cuDNN 5.1 |
| 4   | 2.3 GHz Intel Core i7 | 16 GB 1600 MHz DDR3 | 2473 +/- 34       | 7091 +/- 117               | MacBook Pro Late 2013 / CUDA v7.5 / cuDNN 5.1 |
