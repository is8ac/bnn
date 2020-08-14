# bitnets
This project is currently in development. Do not expect it to work.

In current greedy layer wise training:
- Time is liner with number of parameters and also linear with training set size
- Memory usage is linear with training set size

Memory usage linear with training set size is fine for CIFAR, but will be an issue for ImageNet.




# Literature
## Low precision neural nets

### One bit:
- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)
- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://pjreddie.com/media/files/papers/xnor.pdf)
- [Transfer Learning with Binary Neural Networks](https://arxiv.org/abs/1711.10761)
- [Combinatorial Attacks on Binarized Neural Networks](https://arxiv.org/abs/1810.03538)
- [FINN: A Framework for Fast, Scalable Binarized Neural Network Inference](https://arxiv.org/abs/1612.07119)
- [Probabilistic Binary Neural Networks](https://arxiv.org/abs/1809.03368)
- [A Review of Binarized Neural Networks](https://www.mdpi.com/2079-9292/8/6/661/pdf)

https://openreview.net/pdf?id=ryxfHnCctX

### Ternary:
- [Ternary Weight Networks](https://arxiv.org/abs/1605.04711)
- [Ternary Neural Networks with Fine-Grained Quantization](https://arxiv.org/abs/1705.01462)
- [Ternary Neural Networks for Resource-Efficient AI Applications](https://arxiv.org/abs/1609.00222)
- [Unrolling Ternary Neural Networks](https://arxiv.org/abs/1909.04509)

### Two bit / quaternary:
- [Two-Bit Networks for Deep Learning on Resource-Constrained Embedded Devices](https://arxiv.org/abs/1701.00485)
- [Recursive Binary Neural Network Learning Model with 2-bit/weight Storage Requirement](https://openreview.net/forum?id=rkONG0xAW)

### Mixed
- [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160)

## Genetic / evolutionary optimization
- [Welcoming the Era of Deep Neuroevolution (uber)](https://eng.uber.com/deep-neuroevolution/)
- [Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning](https://arxiv.org/abs/1712.06567)
- [Evolving Deep Neural Networks](https://arxiv.org/abs/1703.00548)
- [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://blog.openai.com/evolution-strategies/)


[AdaComp : Adaptive Residual Gradient Compression for Data-Parallel Distributed Training](https://arxiv.org/abs/1712.02679)

## Clustering
- [Deep Clustering for Unsupervised Learning of Visual Features](https://arxiv.org/abs/1807.05520)
- [Learning Feature Representations with K-means](https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf)
- [Convolutional Clustering for Unsupervised Learning](https://arxiv.org/abs/1511.06241)

# Layerwise training
- https://arxiv.org/abs/1901.06656
- https://arxiv.org/abs/1803.09522
- https://arxiv.org/abs/1812.11446
- https://arxiv.org/abs/1608.05343
- https://arxiv.org/abs/1310.6343
- [A Theory of Local Learning, the Learning Channel, and the Optimality of Backpropagation](https://arxiv.org/abs/1506.06472)
- https://arxiv.org/abs/1905.11786

Loc Quang Trinh's thesis paper: https://dspace.mit.edu/bitstream/handle/1721.1/123128/1128279897-MIT.pdf


[Neural Network with Binary Activations for Efficient Neuromorphic Computing](http://cs229.stanford.edu/proj2016/report/WanLi-NeuralNetworkWithBinaryActivationsForEfficientNeuromorphicComputing-Report.pdf)
