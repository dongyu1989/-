# Attention Is All You Need

## 摘要
目前主要序列转入模型主要是基于复杂的循环或卷积神经网络，这样网络一般是编码器和解码器架构。在这些性能最好的模型之间还通过一个注意机制连接编码器和解码器。我们提出了一种新的简单网络结构即Transformer，它完全基于注意机制，完全不需要递归和卷积。对两个机器翻译任务的实验表明，这些模型在质量上更优，同时更具并行性，训练时间明显更少。对两个机器翻译任务的实验表明，这些模型在质量上更优，同时更具并行性，训练时间明显更少。我们的模型在WMT 2014英语翻译任务中实现了28.4 BLEU，比现有的最佳效果（包括合奏）提高了2倍以上。在WMT2014英语到法语翻译任务中，我们的模型在8个GPU上训练3.5天后建立了一个新的单一模型，即最先进的BLEU分数41.8，这只是文献中最佳模型训练成本的一小部分。结果表明，该Transformer可以很好地推广到其他任务中，并成功地应用于大样本和有限样本的英语用户分析。



## 学习视频
[Attention Is All You Need 论文翻译](https://blog.csdn.net/qq_29695701/article/details/88096455)

[跟李沐学AI](https://www.baidu.com/](https://www.bilibili.com/video/BV1pu411o7BE?spm_id_from=333.337.search-card.all.click&vd_source=97f3f76c3508d1560f7e42c25682c326))

[李宏毅 transformer精讲](https://www.youtube.com/watch?v=ugWDIIOHtPA&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&index=62)

[汉语自然语言处理-从零解读碾压循环神经网络的transformer模型(一)](https://www.bilibili.com/video/BV1P4411F77q?spm_id_from=333.880.my_history.page.click&vd_source=97f3f76c3508d1560f7e42c25682c326)
