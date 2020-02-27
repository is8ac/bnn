# Problem
We want to classify examples into a finite number of classes.
Long term, this may include sequence data as such as log messages.
Short term examples means CIFAR10.
CIFAR10 is a set of 32x32 pixel RGB photographs.
It has 10 classes.

It is well established that muti layer convolution nets are capable of classifying CIFAR10 well.

Traditionally, multilayer conv nets are trained via end to end stochastic gradient decent via back propagation.
We take a minibatch of, perhaps 32 examples, run the foreword pass, caching each intermediate state, to compute the average loss.
Then we compute the gradient of each parameters, making a small update to each weight.

## Problem
This approach has several problems.

### Floating point
End to end back propagation via gradient decent requires continues weights and activations.
Even 32bit float is not good enough for deeper models without mitigations for vanishing gradient.

### Parallelism
The obvious approach to improving the performance of stochastic gradient decent via backpropagation is to distribute the elements of each minibatch across many workers.
This has two limitations:
- Parallelism is limited by number of examples in the minibatch. To parallelize across 1024 workers, we need 1024 examples in each minibatch. This is wasteful of compute. Larger minibatches also tend to produce models which generalize less well.
- Communication latency: Each update, the master must send a copy of the weights to all workers, and then each worker must send its updates back to the master. The weights and updates are many gigabytes for a moderately large model.

### Many updates
The number of updates requires increases with number of layers.
Theoretically worse case, it is exponential with number of layers.
In practice, for a variety of reasons, it it is not nearly so bad, but still, it is superlinear with number of layers.
After great effort, some Google people trained resent50 to 76% in 2500 updates. This was a lot of work. They used a mini-batch size of 65536.


# Goals

## Short term
- Train CIFAR10 to >94% accuracy on a 96 core Cavium ThunderX in less then 10 seconds, and/or for less then $0.02, thereby achieving first place on DAWN bench. Failing that, just getting in the top 5 on time/cost with just 96 arm cores.
- Easily generate a model with binary activations and binary/ternary weights.

## Long term
- Train very large models on billions of images.
- Get results in minutes.
- Distribute this work over spot instances in dozens of data centers across multiple continents.
- All training data stays in its own original data center.
- Communication between training nodes is low bandwidth and high latency.

## Alternative long term goals
- Train a very large 1d conv net on all our data center logs from all data centers while keeping all data local to the data center which produced it so as to comply with data privacy laws.


# Greedy layer-wise training
- [Greedy Layerwise Learning Can Scale to ImageNet](https://arxiv.org/abs/1812.11446)
- [Greedy Layerwise Training of Convolutional NeuralNetworks](https://dspace.mit.edu/bitstream/handle/1721.1/123128/1128279897-MIT.pdf?sequence=1&isAllowed=y)

In late 2018,  Belilovsky et al. said that we can train multi layer conv nets one layer at a time.
For each layer, we construct a simple model consisting of a a single conv layer, followed by global average pooling and a single fully connected auxiliary layer.
This simple model, we train, and then, once it is trained well, we discard the auxiliary layer and apply the conv layer to the images to produce the input for the next layer.
At any given time, we are only solving a single hidden layer problem.
This is nice.
Now number of updates is linear with number of layers.

But while this is a small improvement, it does not grant us all of what we desire.


# Clustering
Consider a big set of 10,000 examples.
Some are of class A, others are of class b.
Consider that the examples are 8 bits in size.
It is wasteful to train on all the examples individually.
There can only be 256 different 8 bit strings.
Far better to allocate 256 countered for each class, and then train on this summary.

Consider instead that each example is 32 bits. Now it would take 4 billion counters per class.
This is still barely doable on modern hardware.
But if each example is 64 bit, it becomes completely imposable.

But consider, the examples are not randomly distributed across the space.
Also, any individual bit is probably not very important by itself.
We can cluster the examples.
As long as each example is within a small hamming distance of a centroid, clustering, and training on the clustered form, will probably incur a negligible cost to accuracy.


It is obviously silly to try to cluster whole images.
A dog looking to the right is an entirety different set of pixels then a dog looking to the left.
32x32 pixels contains too much meaningless information to be worth clustering in there entirety.
Since an end to end trained convolution model depends in the exact positions of all the pixels, we cannot cluster the input when training end to end.

However, a 3x3 pixel patch is far less information.


# Further implications of layer-wise training of conv nets
As described by Belilovsky et al., the input to the auxiliary layer is a 3x3 convolution layer followed by global channel-wise global pooling.
But consider, global average pooling destroys all spatial information.
We could just as easily represent this as an unordered bag of patches.
We multiply each patch in the bag with the convolution filter, apply tanh activation, average the results, and feed into the auxiliary layer.
This does improved our situation.
It increases our memory usage by a factor of 9.

However, now our first layer is a fully connected model in 3x3 pixel patches.
Clustering images is pointless.
Clustering 3x3 patches is very much possible.

Assuming each pixel to be 32 bits, a 3x3 patch consists of 32*3*3=288 bits.
288 bits of a big space.
However, we should not think of the 32 bits of a pixel as being 32 independent bits representing 32 shannons of information.

Rather, they represent 32 different features which are far from independent.
There is much mutual information between bits.
In practice, they tend to be something more like 1 hot, or few hot.

Additionally, the 9 pixels are spatially close and are therefore not independent.

In the early layers, pixels represent simple patters.
There are only so many different edges to be had.
In later layers, pixels represent high level abstract concepts such as "this is a dog", or "this is not a toaster".
This too is a fairly small space.

I hope, based on both theory and tentative evidence, that, in any layer, 3x3 pixel patches can be clustered into a few thousand clusters while loosing negligible information.

Whereas previously an example consisted of a bag of n_pixels patches, now it consists of a far smaller weighed set of pointers to the shared set of patches, where the patch weights sum to n_pixels.

To look at it differently, given k clusters, each image is represented as an array of k counters, summing to n_pixels, many of which will likely be 0 if they never appeared in that image.
It is possible that two entirety different images, for example, one of a dog and another of a toaster, could be represented as nearly identical sets of centroid counts if they happen to contain the same distribution of patches.
(In a well trained mode, the rate of such image level collisions across classes will decrease as we add layers, while the rate of image level coalitions within classes will increase until, if the model is trained to 100% accuracy, all the members of a given class will be represented by the same, or very similar, distributions of centroid counts.)
To compute the input to the auxiliary layer of some set of examples, first we apply the conv filter to each of the k centroids to obtain k sets of activations. This need only be done once for all images.
Then, for each image, we need only elementwise sum the activations, weighting according to the centroid counts of that image.

This makes each image far cheaper.

## Image clustering
Now that each example is a set of k counts, we may additionally cluster images.
This reduces the number of images.

# Training on patch bags
Now that we have reduced the problem to summing the activations of patches in different weightings, and then applying a single fully connected auxiliary layer, we can train.

Evolutionary optimization does not work well with mini-batches and is too expensive for large models.
Now I am attempting to apply backpropagation based gradient decent.
