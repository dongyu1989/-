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

我们对基于注意力的模型的未来感到兴奋，并计划将其应用到其他任务中。我们计划将Transformer扩展到涉及输入和输出模式（文本除外）的问题，并调查本地、受限注意机制，以有效处理图像、音频和视频等大型输入和输出。我们的另一个研究目标是减少一代人的顺序。

我们用来训练和评估模型的代码可以在https://github.com/tensorflow/tensor2tensor上找到。


## 学习视频
[Attention Is All You Need 论文翻译](https://blog.csdn.net/qq_29695701/article/details/88096455)

[跟李沐学AI](https://www.bilibili.com/video/BV1pu411o7BE?spm_id_from=333.337.search-card.all.click&vd_source=97f3f76c3508d1560f7e42c25682c326)

[李宏毅 transformer精讲](https://www.youtube.com/watch?v=ugWDIIOHtPA&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&index=62)

[汉语自然语言处理-从零解读碾压循环神经网络的transformer模型(一)](https://www.bilibili.com/video/BV1P4411F77q?spm_id_from=333.880.my_history.page.click&vd_source=97f3f76c3508d1560f7e42c25682c326)
