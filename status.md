# Current status of generating binary parameters for multi layer image 2d convolution network

# Glossary
## Feature
Dimensions of information about the input.
At the beginning, a single feature might represent an edge.
Later, a single feature might represent higher level concepts such as an dogs eye or a whole dog.
Currently, each feature is stored as one bit, set if the feature is present, unset if not.

## Filter
A filter allows us to extract a feature from an input.
In our usage, it is a bit string of the same length and shape as the input.
We compute the hamming distance between the filter and the input to get the activation, and then compare the activation with the threshold to get the feature.
The threshold is usually either half the number of bits in the input, or the median of the activations of the filter on all the inputs in the training set.

## Split filter
Given two sets of examples, A and B, the filter which has a hamming distance close to A and far from B.
`bitpack(avg(A) - avg(B))`

This is computationally cheap and data parallelizes well.

# Main problem
We need to generate a list of Y X-bit strings such that, given an X bit input, we can generate Y "useful" features from it, where "useful" is some vaguely define notion of containing information which the next layer can use.

There are two main aspects of usefulness:
- Compressing information
- Discarding unimportant information

"linear separability" is one term for part of what we are trying to improve.

## Compression
Given an input, we want to encode it in a few bits while preserving as much information as we can.
This will allow the next layer to use the information in the input more cheaply.
It will also tend to remove noise which may help the model to generalize.

## Discarding unimportant information
Given two different types of example that are both of the same class, we want to encode them as the same embedding.
Information that is not useful for classification should be discarded.
This lossy compression of inputs allows the next layer to ignore unimportant information, allowing it to be both cheaper and generalize better.

# Serial layer generation
The standard way to achieve both of these is to apply end to end back propagation across all the layers of the model.
This works, but has certain drawbacks.
- requires floating point
- parallelizes badly
- Complex loss surface

It would be far nicer of we could generate each layer serially.
[Belilovsky et al.](https://arxiv.org/abs/1812.11446) have demonstrated that it is possible to train each layer serially and achieve better then AlexNet accuracy.
Such a one hidden layer model has a far simpler loss surface.
However training still parallelizes badly and requires floating point.

While trying to replace gradient decent across many layers with some simpler algorithm is unreasonable, trying to replace gradient decent across one hidden layer with a simpler algorithm is more plausible.



# Another mostly unrelated problem: Colors features vs spatial features
For the past few months, one significant issue has been that feature extraction on the first layer focuses excessively on color features.
Why does it do this?

Consider some gray scale patterns.
Compare a set of white to gray edges with a set of gray to black edges.
What is the most significant difference?

One might think that it is the direction of the gradient; that the first is light to dark while the second is dark to light is the most salient difference.
Unfortunately, the model is looking at absolute color values.
It sees that some images are dark while others are light.
It is as if we took our set of features in X and Y dimensions, and randomly noised them in the z dimension.
Since the Z noise is larger then the information in X and Y, the model will focus on the Z dimension to the exclusion of X and Y.
And it is not just Z, but two more dimensions of noise which have been introduced.
In terms of hamming distance, it is color that is the significant difference between examples.
Instead of just gray scale, it has 3 independent channels to deal with.
It must first split by all three independent color channel.
Given just one split per channel, that is 8 different color versions of each spatial feature.

AlexNet appears to learn to make a distinction between color and spatial features due to its two tower architecture.
How single tower deep conv net architecture handle this I do not know.
End to end gradient decent is powerful, it can work around a multitude of problems.

However we do not have the luxury of end to end gradient decent; we must explicitly tell our model to extract spacial features instead of color.
How can we do this?

Consider the case in which we wanted to extract color features and _not_ spatial features.
How would we kill spatial feature while preserving color?
In this case it is simple; we average color across 3x3 patches.
Now all spatial information has been lost within each 3x3 patch, while color is preserved.

We can reverse this.
Instead of averaging color within a 3x3 patch, let us normalize it.
Now, the average color of each 3x3 patch is gray, but the spacial features are preserved.

Now we have killed off all color information between 3x3 patches, leaving the model free to extract spatial features without distraction.

Now we have an image that is flat gray beyond the the 3x3 resolution.
This has probably lost some useful information.
How can we preserve it?

After each block of layers, we spatially reduce by a factor of 2.
We can 2x2 avg pool the original image and reintroduce it to the model before each block.
In this way, high level color information is preserved for usage by later blocks that can actually benefit from it.


# Returning to the main problem
Now that we have dealt with the issue of excessive color features let us return to the main issue: how to generate good features, both features that preserve information and features that discard unimportant information.

## Unsupervised feature extraction
Given some 3x3 pixel patch, we need to embed it into, for example, 32 bits.
Given 45 million such 3x3 patches, we need to create a set of 32 filters that will tell us if the corresponding bit should be set.

This is an unsupervised problem, we are not trying to reduce linear separability between classes, merely compress information with as little loss as we can.

One candidate is k-means clustering.
It is probably not as good as an auto-encoder, but we can approximate it quite cheaply.

## Supervised feature extraction
Eventually we need to discard unimportant information.

Given two sets of examples the distinction between which is unimportant to future layers, we need to find a filter that activates for both sets but not for nonmembers.

Consider a set of 10 classes.
Each class is randomly assigned each of 32 features.
If a feature is assigned to a class, members of that class posses the feature most of the time.
If a feature is not assigned to a class, members of that class rarely posses that feature.
How do we extract these 32 features from the 10 classes?

Let us split the classes into two partitions, A and B.
It it likely that their are some features that were assigned to all of the classes in A and not to classes of set B.
Given a class split, we can extract the features that were assigned differently between A and B.

If we keep splitting the set of classes, and extracting the feature that splits them, we can eventually get most of the 32 features.
They may be mixed together somewhat and/or duplicated, but we have improved there purity.


# Conclusion
We now have two tools to generate features from inputs.
One, is unsupervised and attempts to preserve as much information as it can.
The other is supervised and attempts to extract the dimensions along which the classes differ while discarding unimportant information.
If we stack up many layers of feature extraction, each layer extracting features form 3x3 patches, and 2d convolving the input with that set of feature to produce the next layer, we may get some nice features which can be used to classify the images.

Both feature extraction techniques are amenable to very large MapReduce.
They do not require high bandwidth low latency data transfer.
