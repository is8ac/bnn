# bitnets
This project is currently in development.

To try it out, run
```
cargo run --release --bin mnist_conv
```


# Literature
## Low precision neural nets

### One bit:
- [Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1](https://arxiv.org/abs/1602.02830)
- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://pjreddie.com/media/files/papers/xnor.pdf)
- [Transfer Learning with Binary Neural Networks](https://arxiv.org/abs/1711.10761)

### Ternary:
- [Ternary Weight Networks](https://arxiv.org/abs/1605.04711)
- [Ternary Neural Networks with Fine-Grained Quantization](https://arxiv.org/abs/1705.01462)
- [Ternary Neural Networks for Resource-Efficient AI Applications](https://arxiv.org/abs/1609.00222)

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

Stages:
- For each layer:
  - Allocate an empty vector to hold filter sets
  - For each boosting:
    - Filter the set of patches to those which none of the existing filter sets can correct classify
    - Calculate the average grads for all patches
    - Calculate the average grads for the patches of each label
    - Subtract the global average grads from the label specific average grads
    - Threshold at 0 and bitpack
    - Push the set of bitpacked filters into the vector of filter sets.
  - Flatten the filter set and de-duplicate
  - Take output_chans * 64 filters from the de-duplicated set of filters, and use as the layer filters
  - Calculate the mean of all activations for each filter, and use as the thresholds
  - Use both the layer filters and the thresholds to calculate the bitpacked output of the layer to be used as input to the next layer.
