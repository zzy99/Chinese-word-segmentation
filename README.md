# Chinese-word-segmentation
**中文分词——《自然语言处理导论》项目报告**



**实验目的**

​       编程实现一个LSTM 模型来完成中文自动切词任务，即把连续的中文文本切分成词序列。在测试数据得到测试Precision，Recall，F-score。会提供自动评测F-score 的perl 脚本文件。可以使用提供的脚本文件，也可自己实现相关的评估指标。

 

**数据描述**

•         来自Sighan-2004 中文切词国际比赛的标准数据。

•         训练数据为86924 个句子，大小为16M。

•         测试数据为3985 句子，大小为0.6M。

•         测试数据的答案为test.answer.txt，用于计算系统的Precision，Recall，F-score。

 

**方法**

1.       数据处理

•         数据清洗，去除不匹配的引号

•         按行读入，每行按空格分成一个个词组成的列表

•         给每个字打上标签（如果词长度为1，就是S；为2，就是BE；大于2，就是BM…ME）

•         tokenize，文本序列化，并pad成固定长度（我选择了100）

![](https://i.loli.net/2019/12/30/vbUwCqkLYeWPyxl.png)

​                                                                             句子长度分布

•         将标签onehot，分成五类（B、M、E、S、padding值对应的标签）

2.       加载词向量glove.6B.300d，计算embedding矩阵

3.       搭建模型，架构如图：

![](https://i.loli.net/2019/12/30/yVu283Uoic6DIka.png)

​       输入N个长为100的句子，embedding后用双层双向LSTM得到句子的信息，然后在最后接一个CRF层，用于在整个序列上学习最优的标签序列。我使用了300的embedding dim、100的hidden dim，以及0.5的Dropout。

![](https://i.loli.net/2019/12/30/Wl5BUT29NVd1feI.png)

​                                                                            模型层和参数

4.       训练，验证，测试

 

**实验结果**

​       本次实验采用Google的colab环境，收敛速度很快，训练集上的准确度很快到达了97%以上，然后在测试数据上的F1、Precision、Recall 也能达到95%以上

![](https://i.loli.net/2019/12/30/uGAcjUC2LEerT3H.png)
![](https://i.loli.net/2019/12/30/HOaUbpfBWZ4JP86.png)

评价函数的实现和测试数据上的结果

 

​       然后我把结果直观地和pkuseg进行对比：

![](https://i.loli.net/2019/12/30/6AUuyFfD9cz3biS.png)

​                                                                pkuseg

![](https://i.loli.net/2019/12/30/wmnrHGP6xLY1f3c.png)

​                                                                myseg

![](https://i.loli.net/2019/12/30/wmnrHGP6xLY1f3c.png)

​                                                               标准答案

​       可以发现，在这个数据集上，我们的结果和提供的标准答案几乎没有区别，而pkuseg略有不同，它倾向于把“较小”、“多年来”这些进一步地划分。

 

**总结反思**

​       这次分词作业，我掌握了最简单的序列标注任务流程，也进一步熟练了NLP一些基本操作。如果只针对LSTM模型，结果已经几乎到达最好了，但是还可以思考一些改进，比如参数的调优、4-tag改为6-tag等等，而且我们限制了长度为100，更长的只能切分成几段解决，如果添加mask而不是使用第五个标签，会不会效果更好，值得进一步地尝试。
