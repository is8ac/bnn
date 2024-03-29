# bitnets

See description: https://www.isaacleonard.com/ml/distributed/

```
https://rustup.rs/
```
You will need nightly rust
```
rustup default nightly
```
and make sure it is up to date.
```
rustup update
```
Now you can build it.
```
cargo build --release
```

See how fast different word sizes are on your machine. (Replace "zen+" with the appropriate arch name.)
```
./target/release/count_bits_benchmark 20 zen+ "Threadripper 2950X 16-Core" word_perf_zen+.json
```

Plot the numbers (replace 3000 with a scaling factor appropriate for your machine):
```
./target/release/plot_word_perf 3000 word_perf_zen+.json word_perf_zen+.png
```

And finally stack some layers and measure accuracy on a given training set:
```
./target/release/layer_demo 20 "/path/to_some/big/chunk/of/text.txt"
```

Or run `setup.sh`.

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
- [Binary Neural Networks: A Survey](https://arxiv.org/abs/2004.03333)
- [Enabling Binary Neural Network Training on the Edge](https://arxiv.org/abs/2102.04270)

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

## Layerwise training
- https://arxiv.org/abs/1901.06656
- https://arxiv.org/abs/1803.09522
- https://arxiv.org/abs/1812.11446
- https://arxiv.org/abs/1608.05343
- https://arxiv.org/abs/1310.6343
- [A Theory of Local Learning, the Learning Channel, and the Optimality of Backpropagation](https://arxiv.org/abs/1506.06472)
- https://arxiv.org/abs/1905.11786

Loc Quang Trinh's thesis paper: https://dspace.mit.edu/bitstream/handle/1721.1/123128/1128279897-MIT.pdf


[Neural Network with Binary Activations for Efficient Neuromorphic Computing](http://cs229.stanford.edu/proj2016/report/WanLi-NeuralNetworkWithBinaryActivationsForEfficientNeuromorphicComputing-Report.pdf)

## Gradient compression
- [TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning](https://arxiv.org/abs/1705.07878)
- [Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training](https://arxiv.org/abs/1712.01887)
