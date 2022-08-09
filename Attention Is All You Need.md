# Attention Is All You Need

## Abstract 
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these 
models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 
2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French 
translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small 
fraction of the training costs of the best models from the literature.

## 摘要
目前主要序列转入模型主要是基于复杂的循环或卷积神经网络，这样网络一般是编码器和解码器架构。在这些性能最好的模型之间还通过一个注意机制连接编码器和解码器。我们提出了一种新的简单网络结构即Transformer，它完全基于注意机制，完全不需要递归和卷积。对两个机器翻译任务的实验表明，这些模型在质量上更优，同时更具并行性，训练时间明显更少。对两个机器翻译任务的实验表明，这些模型在质量上更优，同时更具并行性，训练时间明显更少。我们的模型在WMT 2014英语翻译任务中实现了28.4 BLEU，比现有的最佳效果（包括合奏）提高了2倍以上。在WMT2014英语到法语翻译任务中，我们的模型在8个GPU上训练3.5天后建立了一个新的单一模型，即最先进的BLEU分数41.8，这只是文献中最佳模型训练成本的一小部分。结果表明，该Transformer可以很好地推广到其他任务中，并成功地应用于大样本和有限样本的英语用户分析。

## Introduction
Recurrent neural networks, long short-term memory [12] and gated recurrent [7] neural networks in particular, have been firmly established as state of 
the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [29, 2, 5]. Numerous efforts have 
since continued to push the boundaries of recurrent language models and encoder-decoder architectures [31, 21, 13].

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in 
computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t. This 
inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints 
limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [18] and 
conditional computation [26], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, 
however, remains.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of 
dependencies without regard to their distance in the input or output sequences [2, 16]. In all but a few cases [22], however, such attention mechanisms
are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global 
dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation 
quality after being trained for as little as twelve hours on eight P100 GPUs.

## 导言

递归神经网络(RNN)，特别是长短时记忆(LSTM)[13]和门控递归(GRU)[7]神经网络，已经在时间序列模型和转导问题领域，如语言模型、机器翻译[35,2,5]等任务中，已经成为目前最先进的方法。目前，
还有许多努力尝试去继续推动循环语言模型和编解码器架构的应用边界[38,24,15]。

循环模型通常是沿着输入和输出序列的符号位置进行因子计算。将位置与计算时间中的步骤对齐，先前隐藏状态ht−1的函数和位置t的作为输入，生成一系列隐藏状态ht.这种固有的顺序性排除了训练示例中的
并行化,由于内存限制约束批处理，序列化长度变得至关重要。最近一些工作通过因子分解技巧[21]和条件计算[32]显著提高了计算效率，同时在后者的情况下也提高了模型性能。然而，顺序计算的基本约束
仍然存在。

注意力机制已经成为各种任务中引人注目的序列建模和转换模型的组成部分，模型依赖可以不考虑输入或输出序列中的距离的情况[2，19]。然而，在除少数情况外下[27]，这种注意力机制还是与循环网络相结合使用。

在这项工作中，我们提出了Transformer，一个避免了循环的模型架构，它完全依赖一个注意机制来绘制输入和输出之间的全局依赖性。Transformer 允许更大程度的并行化，可以在8个p100 gpu上经过
短短12小时的训练后，在翻译质量上达到一个新的水平。


## Conclusion
In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most 
commonly used in encoder-decoder architectures with multi-headed self-attention. 

For translation tasks, the Transformer can be trained significantly  faster than architectures based on recurrent or convolutional layers. On both WMT 
2014 English-to-German and WMT 2014 English-to-French translation  tasks, we achieve a new state of the art. In the former task our best model 
outperforms even all previously reported ensembles.

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving 
input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such 
as images, audio and video. Making generation less sequential is another research goals of ours. 

The code we used to train and evaluate our models is available at https://github.com/tensorflow/tensor2tensor

## 结论
在这项工作中，我们提出了完全基于注意的第一序列转导模型Transformer，用多头自注意取代了编码器-解码器体系结构中最常用的循环层。

对于翻译任务，Transformer的训练速度明显快于基于循环层或卷积层的架构。在WMT 2014英语到德语和WMT 2014英语到法语的翻译任务中，我们实现了一种新的艺术状态。在上面所做工作中，我们的最佳模型表现甚至超过了之前所有模型。

我们对基于注意力的模型的未来感到兴奋，并计划将其应用到其他任务中。我们计划将Transformer扩展到涉及输入和输出模式而不是文本的问题，并研究局部的、受限制的注意力机制，以有效地处理大量输入和输出，如图像、音频和视频。减少生成的序列化是我们的另一个研究目标。

我们用来训练和评估模型的代码可以在https://github.com/tensorflow/tensor2tensor上找到。


## 学习视频
[Attention Is All You Need 论文翻译 1](https://blog.csdn.net/qq_29695701/article/details/88096455)

[Attention Is All You Need 论文翻译 2](http://t.zoukankan.com/wwj99-p-12156301.html)

[跟李沐学AI](https://www.bilibili.com/video/BV1pu411o7BE?spm_id_from=333.337.search-card.all.click&vd_source=97f3f76c3508d1560f7e42c25682c326)

[李宏毅 transformer精讲](https://www.youtube.com/watch?v=ugWDIIOHtPA&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&index=62)

[汉语自然语言处理-从零解读碾压循环神经网络的transformer模型(一)](https://www.bilibili.com/video/BV1P4411F77q?spm_id_from=333.880.my_history.page.click&vd_source=97f3f76c3508d1560f7e42c25682c326)
