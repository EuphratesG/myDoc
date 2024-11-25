# transformer相关扣细节
1、[为什么 Transformer 需要进行 Multi-head Attention？](https://www.zhihu.com/question/341222779/answer/814111138)
-------------------------------------------------------------------------------------------------------------

可以类比 CNN 中同时使用**多个滤波器**的作用，直观上讲，多头的注意力**有助于网络捕捉到更丰富的特征 / 信息。**Multi-Head 其实不是必须的，去掉一些头效果依然有不错的效果（而且效果下降可能是因为参数量下降），这是因为在头足够的情况下，这些头已经能够有关注位置信息、关注[语法信息](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E8%AF%AD%E6%B3%95%E4%BF%A1%E6%81%AF&zhida_source=entity)、关注罕见词的能力了，再多一些头，无非是一种 enhance 或 noise 而已。

2、[transformer 是如何处理可变长度数据的？](https://www.zhihu.com/question/445895638/answer/1786447741)
-----------------------------------------------------------------------------------------

对不等长的数据，按照最长或者固定长度进行补齐，利用 padding mask 机制，补齐的数据并不参与训练；和 RNN 的 [mask 机制](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=2&q=mask%E6%9C%BA%E5%88%B6&zhida_source=entity)一样。RNN 是通过 time step 方式处理的，transformer 通过计算 attention 得分矩阵处理的。

**3、**[Transformer 为什么 Q 和 K 使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？](https://www.zhihu.com/question/319339652/answer/1617078433)
------------------------------------------------------------------------------------------------------------------------

使用 Q/K/V 不相同可以保证在不同空间进行投影，增强了表达能力，提高了泛化能力。

[transformer 中为什么使用不同的 K 和 Q， 为什么不能使用同一个值？](https://www.zhihu.com/question/319339652)

如果 K 和 Q 使用不同的值的话, A 对 B 的重要程度是 Key(A) * Query(B), 而 B 对 A 的重要程度是 Key(B) * Query(A). 可以看出, A 对 B 的重要程度与 B 对 A 的重要程度是不同的. 然而, 如果 K 和 Q 是一样的, 那么 A 对 B 的重要程度和 B 对 A 的重要程度是一样的.

我们来看看原论文中的公式：

$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V\\$$

Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V\\

这就是我们非常熟悉的 Self-Attention 的计算公式，在这里 Q 和 K 是相同的，但是在真正的应用中，通常会分别给 K 和 Q 乘上[参数矩阵](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E5%8F%82%E6%95%B0%E7%9F%A9%E9%98%B5&zhida_source=entity)，这样公式会变成如下这样（其中两个参数矩阵进行共享的话，就是 reformer 的做法）：

$$Attention(Q,K,V)=softmax(\frac{Q(W_qW^T_k)K^T}{\sqrt{d_k}})V\\$$

Attention(Q,K,V)=softmax(\frac{Q(W_qW^T_k)K^T}{\sqrt{d_k}})V\\

在这里，我们把目光放在 Softmax 上，当 Q 和 K 乘上不同的参数矩阵时，根据 softmax 函数的性质，在给定一组数组成的向量，Softmax 先将这组数的差距拉大（由于 [exp 函数](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=exp%E5%87%BD%E6%95%B0&zhida_source=entity)），然后归一化，它实质做的是一个 soft 版本的 argmax 操作，得到的向量接近一个 [one-hot 向量](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=one-hot%E5%90%91%E9%87%8F&zhida_source=entity)（接近程度根据这组数的数量级有所不同）。这样做保证在不同空间进行投影，增强了表达能力，提高了[泛化能力](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=2&q=%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B&zhida_source=entity)。

如果令 Q 和 K 相同，那么得到的模型大概率会得到一个类似单位矩阵的 attention 矩阵，这样 [self-attention](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=self-attention&zhida_source=entity) 就退化成一个 point-wise 线性映射，对于注意力上的表现力不够强。

4、[transformer 中 multi-head attention 中每个 head 为什么要进行降维？](https://zhuanlan.zhihu.com/p/360144789/ht%3C/i%3Etps://www.zhihu.com/question%3Ci%3E/35%3C/i%3E0369171/answer/1718672303)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

在**不增加时间复杂度**的情况下，同时，借鉴 **CNN 多核**的思想，在**更低的维度**，在**多个独立的[特征空间](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E7%89%B9%E5%BE%81%E7%A9%BA%E9%97%B4&zhida_source=entity)**，**更容易**学习到更丰富的特征信息。

5、Transformer 计算 attention 的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？
-----------------------------------------------------------

为了计算更快。矩阵加法在加法这一块的计算量确实简单，但是作为一个整体计算 attention 的时候相当于一个隐层，整体计算量和[点积](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E7%82%B9%E7%A7%AF&zhida_source=entity)相似。在效果上来说，从实验分析，两者的效果和 dk 相关，dk 越大，加法的效果越显著

6、[为什么在进行 softmax 之前需要对 attention 进行 scaled（为什么除以 dk 的平方根），并使用公式推导进行讲解](https://www.zhihu.com/question/339723385/answer/782509914)
----------------------------------------------------------------------------------------------------------------------------------

**在输入数量较大时，softmax 将几乎全部的[概率分布](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E6%A6%82%E7%8E%87%E5%88%86%E5%B8%83&zhida_source=entity)都分配给了最大值对应的标签**。也就是说**极大的[点积值](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E7%82%B9%E7%A7%AF%E5%80%BC&zhida_source=entity)将整个 softmax 推向梯度平缓区，使得收敛困难**，**梯度消失为 0，造成参数更新困难**。

7、在计算 attention score 的时候如何对 padding 做 mask 操作？
-----------------------------------------------

这里是因为 padding 都是 0，e0=1, 但是 softmax 的函数，也会导致为 padding 的值占全局一定概率，mask 就是让这部分值取无穷小，让他再 softmax 之后基本也为 0，不去影响非 attention socore 的分布

8、[transformer 为什么使用 layer normalization，而不是其他的归一化方法？](https://zhuanlan.zhihu.com/p/360144789/https%3C/i%3E://www.zhihu.com/question/395811291/answer/1257223002)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

_参见：_[transformer 为什么使用 layer normalization，而不是其他的归一化方法？](https://zhuanlan.zhihu.com/p/360144789/https%3C/i%3E://www.zhihu.com/question/395811291/answer/1257223002)

9、[在测试或者预测时，Transformer 里 decoder 为什么还需要 seq mask？](https://zhuanlan.zhihu.com/p/360144789/%3Ci%3Ehttps://www%3C/i%3E.zhihu.c%3Ci%3Eom/%3C/i%3Equestion/369075515/answer/994819222)
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

10、[Transformer 不同 batch 的长度可以不一样吗？还有同一 batch 内为什么需要长度一样？](https://zhuanlan.zhihu.com/p/360144789/ht%3C/i%3Etps://www.zhihu.com/questio%3Ci%3En/4%3C/i%3E39438113/answer/1714391336)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

11、[Transformer 的 Positional embedding 为什么有用？](https://www.zhihu.com/question/385895601/answer/1146997944)
----------------------------------------------------------------------------------------------------------

参见：[Transformer 的 Positional embedding 为什么有用？](https://www.zhihu.com/question/385895601/answer/1146997944)

12、[Transformer 在哪里做了权重共享，为什么可以做权重共享？好处是什么？](https://www.zhihu.com/question/333419099/answer/743341017)
-------------------------------------------------------------------------------------------------------

Transformer 在两个地方进行了权重共享：

**（1）**Encoder 和 Decoder 间的 Embedding 层权重共享；

**（2）**Decoder 中 Embedding 层和 FC 层权重共享。

**对于（1）**，《[Attention is all you need](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=Attention+is+all+you+need&zhida_source=entity)》中 Transformer 被应用在机器翻译任务中，源语言和目标语言是不一样的，但它们可以共用一张大词表，对于两种语言中共同出现的词（比如：数字，标点等等）可以得到更好的表示，而且对于 Encoder 和 Decoder，**嵌入时都只有对应语言的 embedding 会被激活**，因此是可以共用一张词表做权重共享的。

论文中，Transformer 词表用了 bpe 来处理，所以最小的单元是 subword。英语和德语同属[日耳曼语族](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E6%97%A5%E8%80%B3%E6%9B%BC%E8%AF%AD%E6%97%8F&zhida_source=entity)，有很多相同的 subword，可以共享类似的语义。而像中英这样相差较大的语系，语义共享作用可能不会很大。

但是，共用词表会使得词表数量增大，增加 softmax 的计算时间，因此实际使用中是否共享可能要根据情况权衡。

该点参考：[https://www.zhihu.com/question/3334](https://www.zhihu.com/question/333419099/answer/743341017)

**对于（2）**，Embedding 层可以说是通过 onehot 去取到对应的 embedding 向量，FC 层可以说是相反的，通过向量（定义为 x）去得到它可能是某个词的 softmax 概率，取概率最大（贪婪情况下）的作为预测值。

那哪一个会是概率最大的呢？在 FC 层的每一行量级相同的前提下，理论上和 x 相同的那一行对应的点积和 softmax 概率会是最大的（可类比本文问题 1）。

因此，Embedding 层和 FC 层权重共享，[Embedding 层](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=5&q=Embedding%E5%B1%82&zhida_source=entity)中和向量 x 最接近的那一行对应的词，会获得更大的预测概率。实际上，Decoder 中的 **Embedding 层和 FC 层有点像互为[逆过程](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E9%80%86%E8%BF%87%E7%A8%8B&zhida_source=entity)**。

通过这样的权重共享可以减少参数的数量，加快收敛。

但开始我有一个困惑是：Embedding 层参数维度是：(v,d)，FC 层参数维度是：(d,v)，可以直接共享嘛，还是要转置？其中 v 是词表大小，d 是 [embedding 维度](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=embedding%E7%BB%B4%E5%BA%A6&zhida_source=entity)。

查看 pytorch 源码发现真的可以直接共享：

```
fc = nn.Linear(d, v, bias=False)    # Decoder FC层定义
​
weight = Parameter(torch.Tensor(out_features, in_features))   # Linear层权重定义

```

Linear 层的权重定义中，是按照 (out_features, in_features) 顺序来的，实际计算会先将 weight 转置在乘以输入矩阵。所以 FC 层 对应的 Linear 权重维度也是 (v,d)，可以直接共享。

13、transformer 一个 block 中最耗时的部分？
--------------------------------

可以从参数量的角度回答

14、[Transformer 使用 positionencoding 会影响输入 embedding 的原特征吗？](https://zhuanlan.zhihu.com/p/360144789/http%3Ci%3Es://www.zhihu.co%3C/i%3Em/question/350116316/answer/864616018)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

参见：[Transformer 使用 position encoding 会影响输入 embedding 的原特征吗？](https://www.zhihu.com/question/350116316/answer/864616018)

15、不考虑多头的原因，self-attention 中词向量不乘 QKV 参数矩阵，会有什么问题？
--------------------------------------------------

Self-Attention 的核心是**用文本中的其它词来增强目标词的语义表示**，从而更好的利用上下文的信息。

self-attention 中，sequence 中的每个词都会和 sequence 中的每个词做点积去计算相似度，也包括这个词本身。

对于 self-attention，一般会说它的 q=k=v，这里的相等实际上是指它们来自同一个[基础向量](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E5%9F%BA%E7%A1%80%E5%90%91%E9%87%8F&zhida_source=entity)，而在实际计算时，它们是不一样的，因为这三者都是乘了 QKV 参数矩阵的。那如果不乘，每个词对应的 q,k,v 就是完全一样的。

在相同量级的情况下，qi 与 ki 点积的值会是最大的（可以从 “两数和相同的情况下，两数相等对应的积最大” 类比过来）。

那在 softmax 后的加权平均中，该词本身所占的比重将会是最大的，使得其他词的比重很少，无法有效利用上下文信息来增强当前词的语义表示。

而乘以 QKV 参数矩阵，会使得每个词的 q,k,v 都不一样，能很大程度上减轻上述的影响。

当然，QKV 参数矩阵也使得多头，类似于 CNN 中的多核，去捕捉更丰富的特征 / 信息成为可能。

16、**Self-Attention 的时间复杂度是怎么计算的？**
-----------------------------------

Self-Attention 时间复杂度： $O(n^2 \cdot d)$O(n^2 \cdot d) ，这里，n 是序列的长度，d 是 embedding 的维度。

Self-Attention 包括**三个步骤：相似度计算，softmax 和加权平均**，它们分别的时间复杂度是：

相似度计算可以看作大小为 (n,d) 和(d,n)的两个矩阵相乘： $(n,d)*(d,n)=O(n^2 \cdot d)$(n,d)*(d,n)=O(n^2 \cdot d) ，得到一个 (n,n) 的矩阵

softmax 就是直接计算了，时间复杂度为 $O(n^2)$O(n^2)

加权平均可以看作大小为 (n,n) 和(n,d)的两个矩阵相乘： $(n,n)*(n,d)=O(n^2 \cdot d)$(n,n)*(n,d)=O(n^2 \cdot d) ，得到一个 (n,d) 的矩阵

因此，Self-Attention 的时间复杂度是 $O(n^2 \cdot d)$O(n^2 \cdot d) 。

这里再分析一下 Multi-Head Attention，它的作用类似于 CNN 中的多核。

多头的实现不是循环的计算每个头，而是通过 transposes and reshapes，用[矩阵乘法](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95&zhida_source=entity)来完成的。

> In practice, the multi-headed attention are done with transposes and reshapes rather than actual separate tensors. —— 来自 google BERT 源码

Transformer/BERT 中把 **d ，**也就是 hidden_size/embedding_size 这个维度做了 reshape 拆分，可以去看 Google 的 TF [源码](https://link.zhihu.com/?target=https%3A//github.com/google-research/bert)或者上面的 pytorch 源码：

> hidden_size (d) = num_attention_heads (m) * attention_head_size (a)，也即 d=m*a

并将 num_attention_heads 维度 transpose 到前面，使得 Q 和 K 的维度都是 (m,n,a)，这里不考虑 batch 维度。

这样点积可以看作大小为 (m,n,a) 和(m,a,n)的两个张量相乘，得到一个 (m,n,n) 的矩阵，其实就相当于 (n,a) 和(a,n)的两个矩阵相乘，做了 m 次，时间复杂度（感谢评论区指出）是 $O(n^2 \cdot m \cdot a)=O(n^2 \cdot d)$O(n^2 \cdot m \cdot a)=O(n^2 \cdot d) 。

张量乘法时间复杂度分析参见：[矩阵、张量乘法的时间复杂度分析](https://link.zhihu.com/?target=https%3A//liwt31.github.io/2018/10/12/mul-complexity/)

因此 Multi-Head Attention 时间复杂度也是 $O(n^2 \cdot d)$O(n^2 \cdot d) ，复杂度相较单头并没有变化，主要还是 transposes and reshapes 的操作，相当于把一个大矩阵相乘变成了多个小矩阵的相乘。

**17、Transformer 的点积模型做缩放的原因是什么？**
----------------------------------

参考：[https://www.zhihu.com/question/339723385](https://www.zhihu.com/question/339723385)

18、大概讲一下 Transformer 的 Encoder 模块？
----------------------------------

参见：[DASOU：答案解析 (2)-3 分钟彻底掌握 Transformer 的 Encoder—满满干货！！](https://zhuanlan.zhihu.com/p/151586285)

19、为何在获取输入词向量之后需要对矩阵乘以 embedding size 的开方？意义是什么？
------------------------------------------------

```
embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                           dtype=d_type, name="{}_embeddings".format(name))(inputs)
embeddings *= tf.math.sqrt(x=tf.cast(x=embedding_dim, dtype=d_type), name="{}_sqrt".format(name))

```

使用 Keras 的 Embedding，其初始化方式是 xavier init，而这种方式的方差是 1/embedding size，因此乘以 embedding size 的开方使得 embedding matrix 的方差是 1，在这个 scale 下可能更有利于 embedding matrix 的收敛。

20、简单介绍一下 Transformer 的位置编码？有什么意义和优缺点？
--------------------------------------

参见：ref="[https://z](https://link.zhihu.com/?target=https%3A//z)_huan_[http://lan.zhihu.com/p/106644634](http://lan.zhihu.com/p/106644634)"> 一文读懂 Transformer 模型的位置编码

21、你还了解哪些关于位置编码的技术，各自的优缺点是什么？
-----------------------------

参见：[如何优雅地编码文本中的位置信息？三种 positionalencoding 方法简述](https://link.zhihu.com/?target=https%3A//zhuanlan.zh%253Ci%253Eihu.com/%253C/i%253Ep/121126531)

[让研究人员绞尽脑汁的 Transformer 位置编码](https://zhuanlan.zhihu.com/p/360144789/h%3Ci%3Ettps://zhuanlan%3C/i%3E.zhihu.com/p/352898810)

22、简单讲一下 Transformer 中的残差结构以及意义。
--------------------------------

防止梯度消失，帮助深层网络训练

23、为什么 transformer 块使用 LayerNorm 而不是 BatchNorm？LayerNorm 在 Transformer 的位置是哪里？
------------------------------------------------------------------------------

_参见：_[transformer 为什么使用 layer normalization，而不是其他的归一化方法？](https://zhuanlan.zhihu.com/p/360144789/https%3C/i%3E://www.zhihu.com/question/395811291/answer/1257223002)

24、简答讲一下 BatchNorm 技术，以及它的优缺点。
------------------------------

参见：[https://zhuanlan.zhihu.com/p/153183322](https://zhuanlan.zhihu.com/p/153183322)  
BN 的理解重点在于它是针对整个 Batch 中的样本在同一维度特征在做处理。  
在 MLP 中，比如我们有 10 行 5 列数据。5 列代表特征，10 行代表 10 个样本。是对第一个特征这一列（对应 10 个样本）做一次处理，第二个特征（同样是一列）做一次处理，依次类推。  
在 CNN 中扩展，我们的数据是 N·C·H·W。其中 N 为样本数量也就是 batch_size，C 为通道数，H 为高，W 为宽，BN 保留 C 通道数，在 N,H,W 上做操作。比如说把第一个样本的第一个通道的数据，第二个样本第一个通道的数据..... 第 N 个样本第一个通道的数据作为原始数据，处理得到相应的均值和方差。  
**BN 有两个优点。**  
第一个就是可以解决内部[协变量](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E5%8D%8F%E5%8F%98%E9%87%8F&zhida_source=entity)偏移，简单来说训练过程中，各层分布不同，增大了学习难度，BN 缓解了这个问题。当然后来也有论文证明 BN 有作用和这个没关系，而是可以使损失平面更加的平滑，从而加快的[收敛速度](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E6%94%B6%E6%95%9B%E9%80%9F%E5%BA%A6&zhida_source=entity)。  
第二个优点就是缓解了梯度饱和问题（如果使用 sigmoid [激活函数](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0&zhida_source=entity)的话），加快收敛。  
**BN 的缺点：**  
第一个，batch_size 较小的时候，效果差。这一点很容易理解。BN 的过程，使用 整个 batch 中样本的均值和方差来模拟全部数据的均值和方差，在 batch_size 较小的时候，效果肯定不好。  
第二个缺点就是 BN 在 RNN 中效果比较差。这一点和第一点原因很类似，不过我单挑出来说。  
首先我们要意识到一点，就是 RNN 的输入是长度是动态的，就是说每个样本的长度是不一样的。  
举个最简单的例子，比如 batch_size 为 10，也就是我有 10 个样本，其中 9 个样本长度为 5，第 10 个样本长度为 20。  
那么问题来了，前五个单词的均值和方差都可以在这个 batch 中求出来从而模型真实均值和方差。但是第 6 个单词到底 20 个单词怎么办？  
只用这一个样本进行模型的话，不就是回到了第一点，batch 太小，导致效果很差。  
第三个缺点就是在测试阶段的问题，分三部分说。  
首先测试的时候，我们可以在队列里拉一个 batch 进去进行计算，但是也有情况是来一个必须尽快出来一个，也就是 batch 为 1，这个时候均值和方差怎么办？  
这个一般是在训练的时候就把均值和方差保存下来，测试的时候直接用就可以。那么选取效果好的均值和方差就是个问题。  
其次在测试的时候，遇到一个样本长度为 1000 的样本，在训练的时候最大长度为 600，那么后面 400 个单词的均值和方差在训练数据没碰到过，这个时候怎么办？  
这个问题我们一般是在数据处理的时候就会做截断。  
还有一个问题就是就是[训练集](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E8%AE%AD%E7%BB%83%E9%9B%86&zhida_source=entity)和测试集的均值和方差相差比较大，那么训练集的均值和方差就不能很好的反应你测试数据特性，效果就回差。这个时候就和你的数据处理有关系了。

25、简单描述一下 Transformer 中的[前馈神经网络](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C&zhida_source=entity)？使用了什么激活函数？相关优缺点？
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

前馈神经网络采用了两个[线性变换](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E7%BA%BF%E6%80%A7%E5%8F%98%E6%8D%A2&zhida_source=entity)，激活函数为 Relu，公式如下：

$FFN(x) = max(0, xW_1 + b_1) W_2 + b_2$ FFN(x) = max(0, xW_1 + b_1) W_2 + b_2

优点：

SGD 算法的收敛速度比 sigmoid 和 tanh 快；（梯度不会饱和，解决了[梯度消失问题](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E9%97%AE%E9%A2%98&zhida_source=entity)）

计算复杂度低，不需要进行指数运算；

适合用于后向传播。

缺点：

ReLU 的输出不是 zero-centered；

ReLU 在训练的时候很” 脆弱”，一不小心有可能导致神经元” 坏死”。举个例子：由于 ReLU 在 x<0 时梯度为 0，这样就导致负的梯度在这个 ReLU 被置零，而且这个神经元有可能再也不会被任何数据激活。如果这个情况发生了，那么这个神经元之后的梯度就永远是 0 了，也就是 ReLU 神经元坏死了，不再对任何数据有所响应。实际操作中，如果你的 learning rate 很大，那么很有可能你网络中的 40% 的神经元都坏死了。 当然，如果你设置了一个合适的较小的 learning rate，这个问题发生的情况其实也不会太频繁。，Dead ReLU Problem（神经元坏死现象）：某些神经元可能永远不会被激活，导致相应参数永远不会被更新（在负数部分，梯度为 0）。产生这种现象的两个原因：参数初始化问题；learning rate 太高导致在训练过程中参数更新太大。 解决方法：采用 Xavier 初始化方法，以及避免将 learning rate 设置太大或使用 adagrad 等自动调节 learning rate 的算法。

ReLU 不会对数据做幅度压缩，所以数据的幅度会随着模型层数的增加不断扩张。

26、Encoder 端和 Decoder 端是如何进行交互的？（在这里可以问一下关于 seq2seq 的 attention 知识）
-------------------------------------------------------------------

参见：[ransformer 中的细节探讨](https://zhuanlan.zhihu.com/p/360144789/htt%3C/i%3Eps://zhuanlan.zhihu.com/p/58969651)

27、Decoder 阶段的多头自注意力和 encoder 的多头自注意力有什么区别？（为什么需要 decoder 自注意力需要进行 sequence mask)
---------------------------------------------------------------------------------

28、Transformer 的[并行化](https://zhida.zhihu.com/search?content_id=168161579&content_type=Article&match_order=1&q=%E5%B9%B6%E8%A1%8C%E5%8C%96&zhida_source=entity)提现在哪个地方？Decoder 端可以做并行化吗？
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

29、简单描述一下 wordpiece model 和 byte pair encoding，有实际应用过吗？
-------------------------------------------------------

参见：[深入理解 NLP Subword 算法：BPE、WordPiece、ULM](https://link.zhihu.com/?target=https%3A//zhuan%253Ci%253Elan.zhihu%253C/i%253E.com/p/86965595)

30、Transformer 训练的时候学习率是如何设定的？Dropout 是如何设定的，位置在哪里？Dropout 在测试的需要有什么需要注意的吗？
---------------------------------------------------------------------------

31、[有哪些令你印象深刻的魔改 transformer？](https://zhuanlan.zhihu.com/p/360144789/htt%3Ci%3Eps://www.zh%3C/i%3Eihu.com/question/349958732/answer/945349902)
-----------------------------------------------------------------------------------------------------------------------------------------------

