---
layout: post
title:  "CV Series #1: SqueezeNet"
date:   2019-10-20 00:00:00 -0700
categories: [cv]
---
<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
For my first blog post, I'm going to talk about SqueezeNet (*Iandola et al, 2016* [[1]](https://arxiv.org/pdf/1602.07360.pdf)), a CNN architecture developed in 2016 that brought compact neural network design to the forefront of machine learning. I chose to discuss SqueezeNet because:
* It was immensely influential (~ 1640 citations)
* It brings together several ideas that are now key parts of the broader field of *efficient machine learning* (an area I think is fascinating and very relevant today). 

# **What problem does SqueezeNet try to solve?**

To understand the problem SqueezeNet tries to solve, we need to (mentally) go back in time to 2016.

Neural networks were only getting bigger and bigger; ResNet (*He et al, 2015*) had allowed us to surpass the 100-layer barrier, and state-of-the-art architectures could easily surpass a **100 million** parameters (VGG was ~130 million, AlexNet was 60 million). This also meant, however, that the computational footprints of networks like these was also growing larger and larger, which is unsuitable for many reasons:

* *Memory-based issues*:

As we can see below, traditional architectures like AlexNet (*Krizhevsky et al, 2012*) are far too resource-heavy to run inference on edge or embedded devices, which are becoming increasingly important in areas like IoT (Internet of Things)! [[2]](https://arxiv.org/pdf/1905.12107.pdf)[[3]](https://arxiv.org/pdf/1602.01616.pdf)

<img src="/assets/SqueezeNet_diag1.jpg" alt="fig_1" width="700"/>



* *Commmunication overheads*:

As we can see below in the example of Tesla, communicating model updates to clients can get staggeringly expensive with larger model sizes; this problem also applies to distributed training, where the size of the gradient (which is communicated between nodes at each step) is **proportional** to the model size.

<img src="/assets/SqueezeNet_diag2.jpg" alt="fig_2" width="700"/>

**SqueezeNet** was developed to mitigate these problems, while *maintaining the accuracy* of state-of-the-art models like AlexNet.

# **Context: what inspired some of the ideas in SqueezeNet?**

*If I have seen further, it is by standing on the shoulders of giants." - Isaac Newton* <img src="/assets/newton.png" alt="Newton" width="100"/>


As we shall soon discuss, the SqueezeNet paper contains some very interesting ideas. However, while reading the paper, I realized that those ideas became even richer when understood in the context of two papers that came before it - GoogLeNet (Szegedy et al, 2014) and ResNet (He et al, 2015)


#### (1) *GoogLeNet (Szegedy et al, 2014)*
GoogLeNet [[7]](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf), in some ways, is best understood as one of a pair, the other being VGGNet [[6]](https://arxiv.org/pdf/1409.1556.pdf). Both were developed in 2014, focused on increasing the depth of neural networks, and won several challenges at the ILSVC (ImageNet Large Scale Vision Challenge) that year, two years after AlexNet had taken the same challenge by storm.

VGGNet used a combination of:
*    Smaller kernel sizes (3x3, vs the 11x11s seen in the first layer of AlexNet)
*    Sandwiching multiple convolutional layers in between pooling layers

These strategies enabled them to train the deeper network better, but also increased the parameter count even further than AlexNet (from 60 million to **more than 130 million**!)


In contrast, GoogLeNet took a radically different approach to VGG, with a *conscious* goal of keeping the parameter count and computational overhead in check, as opposed to throwing more data, hardware and layers at the problem.

<img src="/assets/SqueezeNet_diag5.png" alt="inception" width="900">

The principle feature of GoogLeNet was the Inception module (depicted above), and was driven by two key ideas:
  * Inspired to some extent by the fact that primate visual cortexes seem to use a variety of scales, the Inception module used different kernel sizes in the same layer (3x3, 5x5 and 1x1), as shown in (a).


  * The module also used 1x1 convolutions before these layers (as shown in (b)) to reduce the dimensionality/number of channels fed into the 3x3 and 5x5 convolutions.

These Inception modules, as highlighted on the right-hand side of the diagram above, were then used as building blocks for the GoogLeNet network; this idea of modular design is something we will return to when discussing SqueezeNet itself.

#### (2) *ResNet (He at al, 2015)*

While VGGNet and GoogLeNet allowed us to develop deeper networks than before, it was still a significant challenge to train deep neural networks well; deeper networks often ended up with *worse* performances than shallower variants, even though mathematically this made little sense.

To see why, note that if you have a k-layer network $$L_k$$ and a k+1 layer network $$L_{k+1}$$, $$L_{k+1}$$ can easily attain the same performance as $$L_{k}$$ by:
*   Replicating the first k layers of $$L_{k}$$ ($$L_{k+1}[1:k] = L_{k}$$)
*   Adding an identity-like mapping at the end! ($$L_{k+1}[k+1] \approx I$$)

In practice, deeper neural networks were unable to learn good mappings (note - this is **not** simply due the vanishing gradient problem, but extends to a more general inability to optimize deeper models); to solve this, *He et al* proposed a very simple idea, as shown in the diagram below. 

<img src='/assets/ResNet_block.png' alt='ResNet_block'/>

Normally, in neural networks, we want each layer to learn a function $$H_{i}(x)$$, such that the final network is $$N(x) = H_n(H_{n-1}(...H_1(H_0(x))))$$. However, by adding in a residual connection between "layers" (in this case, a block of 2 convolutional layers), you could instead try and learn a new function at each layer, $$F_{i}(x) = H_{i}(x) - x$$ (the output of the layer is still the same, $$F_{i}(x) + x = H_{i}(x)$$)

This now meant that, at the very least, you could easily learn the identity mapping between layers (by squeezing the residual $$F_{i}(x)$$ to 0). The residual connections also acted as a preconditioning on the optimization problem with respect to the weights, by encouraging layers to find mappings closer to the identity. 

<img src='/assets/ResNet_bottleneck_block.png' alt='ResNet_bottleneck'/>

For deeper versions of ResNets, He et al also developed a variant on the residual block, as shown above; the "bottleneck" building block used 1x1 convolutions, in GoogLeNet style, to:
*   Shrink the number of channels fed into the larger 3x3 convolutions (such as from 256 to 64)

*   Expand the result of the 3x3 convolutions back to 256 channels, which are then combined with the original 256-d input via the residual connection


The results from ResNets, using both kinds of blocks, were simply staggering - we were now able to train networks hundreds, or even thousands of layers deep, that clearly outperformed state-of-the-art networks like GoogLeNet or VGGNet! *He et al* also found that many of the layers in their ResNets were indeed close-to-identity mappings; this gave us a clear indication that the kind of residual preconditioning introduced in ResNets could help train all kinds of neural networks.

# **SqueezeNet** - the ideas, models and results
Given our discussion of the problem of large models and the context of both GoogLeNet and ResNet, we're now ready to get into the details of SqueezeNet itself!

#### *Core Strategy*:
SqueezeNet's primary goal was to design CNNs that were much smaller, but still achieved comparable to *state-of-the-art* performance; to do so, the authors outlined a threefold strategy:

* Replace most 3x3 filters with 1x1 filters, where possible; this is a fairly straightforward means to reduce the parameter count *~9x* by making the convolutional kernels smaller.

* Reduce the number of input channels to 3x3 filters. For a 3x3 conv layer, the number of parameters is $$\text{channels}_{\text{input}} \times \text{channels}_{\text{output}} \times (3 \times 3)$$, so reducing the input channels to a 3x3 layer also reduces **the number of parameters** in that layer, an idea clearly influenced by ResNet and GoogLeNet.

* Delay downsampling in the network to improve accuracy (this was based on a result from another work from Kaiming He, the first author of ResNet, and Jian Sun, titled *Convolutional Neural Networks at Constrained Time Cost* [[9]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/He_Convolutional_Neural_Networks_2015_CVPR_paper.pdf). Amongst other things, the paper showed that networks which delayed downsampling layers (such as pooling layers, or convolutional layers with strides > 1) often achieved better accuracy by keeping the activation maps large for a bigger part of the network. 

#### *The Fire module, and its use in building SqueezeNet*:
<img src="/assets/fire_module.png" alt="fire_module" width="300" />

With this strategy in mind, the authors introduced their own building block (just like the Inception module from GoogLeNet) - the Fire module, as shown above. The idea behind the Fire module is fairly derivative from what we've seen in ResNet and/or GoogLeNet - the module employs a smaller number of 1x1 convolutions to cheaply cut the number of input channels to the larger 3x3 filters. 

What's unique about the Fire module, however, is that unlike ResNet (which uses a block of 3x3 filters, **followed by** a block of 1x1 filters to expand the output of the "squeeze layer"), the Fire module *does not re-expand the output of the 3x3s*. Instead, it uses only one mixed block of 3x3s and 1x1s to do a simultaneous expansion of channels **and** receptive field. 

Furthermore, unlike the Inception module, the Fire module is \noticeably much smaller in size and complexity as a block, with only two kinds of filters mixed in; the Fire module also eschews 5x5 filters in favour of mostly 1x1 convolutions (with some 3x3s added in), so as to reduce the complexity and number of parameters per block. 

Using the Fire module as a building block, the authors designed the SqueezeNet family of architectures (as shown in the diagram below). Important things to note are:
* Apart from the first and last layers, the entire network is built out of Fire modules
* The authors experimented with three increasingly complex variants
    * *v1*: A vanilla variant (with Fire modules replacing standard convolutional layers)
    * *v2*: A variant with residual connections between layers with the same number of output channels
    * *v3*: A variant with added connections (via 1x1 convolutions, used to adjust the number of channels) between "mismatched layers", in addition to the pre-existing connections from v2

<img src="/assets/SqueezeNet_diag6.jpg" alt="SqueezeNet" width="900">

#### *Results*
<img src="/assets/squeezenet_results.png" alt="SqueezeNet_Results" width="700">

Above is the table of results from the SqueezeNet paper; as we can see, the authors do not just compare to vanilla AlexNet, but extend that comparison to compressed variants of both AlexNet and SqueezeNet. This ties into the fact that the purpose of SqueezeNet is develop a small model, and that it is therefore essential to integrate the element of compact model design with other available techniques from model compression.

What the results show, in my opinion, is truly amazing - the best SqueezeNet *vanilla* model (6 bit, compressed via Deep Compression[[10]](https://arxiv.org/pdf/1510.00149.pdf)) achieves the same/better accuracies on ImageNet, but is **510x** smaller than vanilla AlexNet! Even when comparing uncompressed to uncompressed, the SqueezeNet model is **50x** smaller than AlexNet, while still achieving the same results . 

It's important to note that other, state-of-the-art networks (especially at the time) reported much "higher" accuracies on ImageNet (for instance, GoogLeNet had a single model top-5 accuracy rate of 92.11%); however, most of these networks used multiple crops of the original image during testing, and averaged the result of their network over these crops (GoogLeNet, for instance, used **144** crops for each image).

When adjusted for single crop testing, SqueezeNet's best model (*with residual connections*) attains 85.3% top-5 accuracy rates, while GoogLeNet attains 88.9% [[11]](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) (both accuracies are reported for the ImageNet *validation* set), which given the significant difference in model size seems like a reasonable gap. (Quick side-note: top-5 accuracy indicates the percentage of times the target class for an image was in **top 5** guesses of the model)

It's also worth nothing that the authors used AlexNet as their baseline because it was the most prevalent one in *the model compression* community; as you can see in the results table, the authors also compare SqueezeNet to other previous compression works (SVD, Network Pruning, Deep Compression), which were all evaluated on AlexNet.

This illustrates the key point of *Iandola et al* - by using clever model design, they were able to attain even better results than most compression methods, showing us that model design was just as (if not even more) important to attaining smaller models.

#### *A quick detour into Deep Compression*
Since the SqueezeNet results are closely intertwined with model compression (and in particular the Deep Compression method), it's worth having a quick look at what Deep Compression actually involves.

<img src="/assets/deep_compression.png" alt="Deep Compression" height="250" width="600">

Deep Compression is a method for compressing neural networks into smaller models (both in terms of parameter size and actual memory required to store the model), and uses three techniques, as the diagram above shows:

* *Pruning*: The idea behind this is to prune weights/connections with small *absolute* values (below a defined threshold), since they do not affect the output of the overall layer as much. This allows us to store the remaining weights in a sparse matrix-style format, and retrain them (to account for the loss of the pruned parameters)

* *Trained Quantization*: The next step in the pipeline is to *quantize* the weights; the idea here is to reduce the number of bits required to **represent** each weight, by having connections share weights (the number of bits you quantize to determines the maxmimum number of allowed shared bits e.g. 8 bits => 256 shared weights). 

This is done by grouping the original weights into bins based on their values, and using the number for each bin as the quantized represention. During the forward/backward pass, we simply use a lookup table to convert the quantized representation to the actual weight (the centroid of the weights that were binned together). As shown in the diagram below,  the quantized weights are retrained as groups (hence the name "Trained Quantization")

<img src="/assets/trained_quantization.png">



* *Huffman Coding*: As shown in the picture below [[12]](http://robotics.cs.tamu.edu/dshell/cs314/sa6/sa6.html) (for a string input, although the same idea can be applied to all kinds of data), Huffman coding is a frequency-based *lossless* encoding scheme. As we can see in the example tree shown below, more frequent "characters" have shorter Huffman codes/paths in the tree which makes it a good scheme to compress data.

<img src="/assets/huffman.png">

The combination of all of these methods allows the Deep Compression method to achieve up to 50x reduction in stored model size **without loss in accuracy**, making it a key benchmark in the model compression community; the fact that SqueezeNet could achieve even lower model size compression rates than Deep Compressed AlexNet makes its results all the more impressive!


### *A study of the SqueezeNet design space*
In addition to the model, the authors also did a study of the design space within the SqueezeNet/Fire module-based family of networks, which is definitely worth going over. The study they conduct is on two levels:

#### *Microarchitectural (the internals of the Fire module)*
<img src="/assets/fire_module.png" alt="fire_module" width="300" />

When we look at the Fire module again (shown above), there are three parameters we can play around with: $$s_{1x1}$$ (The number of 1x1 filters in the squeeze layers) and $$e_{1x1}$$/$$e_{3x3}$$ (The number of 1x1 / 3x3 filters in the expand layer respectively).

Using these, the authors did two experiments (shown below): one involving the *Squeeze Ratio* (the ratio of $$s_{1x1}: e_{1x1} + e_{3x3}$$), and another on the *percentage of 3x3 filters* in the Fire module. 

<img src="/assets/squeezenet_micro.png">

As we can see, increasing the squeeze ratio leads to an increase in accuracy; however, this also (as expected, given the implied increase in channels fed into 3x3 filters and the consequent increase in the layer size) increases the model size significantly! Similarly, increasing the ratio of 3x3 filters in the Fire module does increase the accuracy (up to around the 50% mark, beyond which we see diminishing returns), but as with Squeeze Ratio, this is antithetical to the original goal of smaller model sizes.

In practice, this might suggest trading off either parameter for higher accuracy, subject to the memory requirements of the system; it is encouraging, however, to note that both quantities have clear points where the accuracy returns diminish and are yet superior to other approaches. 


#### *Macroarchitectural (Network design choices):*

As we saw in the SqueezeNet design, there were three variants - Vanilla SqueezeNet, SqueezeNet with residual connections (when layers have the same input and output sizes), and SqueezeNet with complex bypass connections via 1x1 convolutions to account for the differences in input/output sizes for some layers.

The original results table was presented for the Vanilla version - to some degree, this helped ensure a fair comparison given that AlexNet was developed well before residual connections were used. When we add in these variants, we get the following results:

<img src="/assets/squeezenet_results_2.png"/>

We can see some pretty interesting results; while simple bypass connections improve both top-1 and top-5 accuracies on ImageNet without changing model size, complex bypass connections are actually **inferior** to simple bypass and are not much better than Vanilla SqueezeNet.

Complex bypass connections also add significant memory to the model size, making it clear that using simple bypass-based SqueezeNet is the best route. In the same vein as ResNet, this is somewhat surprising (given that the complex bypass connections could simply learn a close-to-zero mapping in theory). However, a reasonable explanation might lie in the fact that complex bypass connections increase the number of parameters to train, but the other training settings are kept the same (epochs trained for, data augmentation, etc). 



### **Impact** - what happened afterwards?
As I mentioned at the beginning, one of the reasons I chose SqueezeNet as the topic for my first blog post was the extent of influence it has had over the efficient ML community. The paper has received over ~1600 citations and was one of the first papers to prove that designing the model for compactness was just as important as any of the other areas being looked into to reduce model size/impact (quantization, compression, sparsity amongst many others).

SqueezeNet has had numerous successors in the compact design space - from direct descendants, like SqueezeNext and SqueezeSeg (developed by the same research group), to architectures like the MobileNets (v1 through v3) that have allowed us to deploy more kinds of neural networks directly onto edge/embedded and mobile devices. It also sparked the creation of DeepScale, a startup that works on efficient deep neural networks and was recently acquired by Tesla!

The paper remains a key read for anyone interested in efficient ML models, and I highly encourage anyone who liked this blog post to check out the original paper, along with some of the papers cited in this article. I used them as a base material for this post, but there's a lot more to them that I've had to leave out (for the sake of compactness).

## Citations
* [[1]](https://arxiv.org/pdf/1602.07360.pdf) *SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size*: Iandola et al, 2016
* [[2]](https://arxiv.org/pdf/1905.12107.pdf) *FPGA-based implementation of Deep Neural Networks using on-chip memory only*: Park et al, 2016 
* [[3]](https://arxiv.org/pdf/1602.01616.pdf) *SpArSe: Sparse Architecture Search for CNNs on Resource-Constrained Microcontrollers*: Federov et al, 2019 
* [[4]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) *ImageNet Classification with Deep Convolutional Neural Networks*: Krizhevsky et al, 2012
* [[5]](https://pixabay.com/vectors/isaac-newton-portrait-vintage-3936704/) *Image of Isaac Netwon by Gordon Johnson from Pixabay*
* [[6]](https://arxiv.org/pdf/1409.1556.pdf) *Very Deep Convolutional Networks For Large-Scale Image Recognition*: Simonyan et al, 2014 
* [[7]](https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf) *Going Deeper with Convolutions*: Szegedy et al, 2014 
* [[8]](https://arxiv.org/pdf/1512.03385.pdf) *Deep Residual Learning for Image Recognition*: He et al, 2015
* [[9]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/He_Convolutional_Neural_Networks_2015_CVPR_paper.pdf) *Convolutional Neural Networks at Constrained Time Cost*: He et Sun, 2015
* [[10]](https://arxiv.org/pdf/1510.00149.pdf) *Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding*: Han et al, 2015
* [[11]](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet) *Berkeley AI Research (formerly the Berkeley Vision and Learning Center)'s re-implementation of GoogLeNet on Caffe*
* [[12]](http://robotics.cs.tamu.edu/dshell/cs314/sa6/sa6.html) *Texas A&M University, Department of Computer Science and Engineering - CSCE 314: Programming Languages*