一些必要的前置知识
---------

统计学，概率论

线性代数（矩阵计算那一块）

离散数学的基本知识（各种符号，极限，级数等得看得懂）

机器学习基本概念
--------

这个就很简单了，随便一搜就是一堆文章，先系统性的对机器学习有个大概得认知就可以了，本节你需要了解：

*   基本模型有啥，是干什么的
*   各个模型为了解决什么东西
*   机器学习的训练测试验证集合是什么，怎么用的
*   机器学习的输入是什么？输出是什么？
*   模型训练是在干什么事？
*   为什么机器学习是在” 训练 “模型
*   什么是欠拟合和过拟合
*   机器学习的训练目标是什么
*   监督无监督和强化是什么

学习这些不用去定量了解，知道这些概念是干啥的就行，如果从这里开始定量了解那基本就停不下来了

这里随便推[一篇](https://zhuanlan.zhihu.com/p/74673610)

我会怎么展开接下来的内容
------------

在了解完机器学习的基本概念之后，你大概也会对机器学习的一些点有了认知（如果以下的点你不了解，也需要去定性地学习下），由于本文只涉及 LLM 相关，因此接下来的内容也将从 nlp 的这些点切入：

预处理（preprocess） -> 分词（Tokenization） -> 模型优化（optimization） -> 模型（会拿 transformer 举例） -> 精调（[fine tuning](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=fine+tuning&zhida_source=entity)）

**那么，接下来咱们从机器学习的整个流程开始逐个切入他们的原理**

预处理（preprocess)
---------------

机器学习极其依赖数据的质量，因此在训练前数据的预处理永远都是最关键的一步，本节你需要了解：

*   预处理做了什么？
*   C4、CommonCrawl 等开源数据源是什么，如果有条件可以下载一些浏览看看
*   这些数据是如何应用到模型的训练中的

需要去看的[文档](https://zhuanlan.zhihu.com/p/619241179)

分词（Tokenization）
----------------

分词，简而言之就是把一句话分成多个词组，例如`我是一个人类，我喜欢水果`可以被分词为`["我", "是", "一个", "人类，", "我", "喜欢", "水果"]`。

由于机器是无法直接理解人类语言的，因此人类语言首先要被转化为机器可以理解的**向量**（这个向量维度越高，所包含的信息也就越多），而在自然语言，词是最小的单位，如果直接把一整句话转化为向量便会丢失大量的信息，所以我们需要首先把句子拆分为词组，然后再对每个词转化为向量，最后将一句话的所有向量拼接成**矩阵**（例如一个句子分词后有 N 个词，每个词的向量维度为 M，那么就可以组成一个 N*M 的矩阵）

本结你需要了解：

*   分词干了什么事？
*   为什么需要把句子分词？
*   bpe、wordpiece 和 unigram language model 这三种分词模型的原理是什么？
*   什么是词嵌入
*   如何解决不同长度句子的转化为矩阵后的维度不一致的问题？

需要去看的文档：

[词嵌入是什么](https://link.zhihu.com/?target=https%3A//easyaitech.medium.com/%25E4%25B8%2580%25E6%2596%2587%25E7%259C%258B%25E6%2587%2582%25E8%25AF%258D%25E5%25B5%258C%25E5%2585%25A5word-embedding-2%25E7%25A7%258D%25E7%25AE%2597%25E6%25B3%2595-%25E5%2585%25B6%25E4%25BB%2596%25E6%2596%2587%25E6%259C%25AC%25E8%25A1%25A8%25E7%25A4%25BA%25E6%25AF%2594%25E8%25BE%2583-c7dd8e4524db)

[byte pair encoding 分词](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/zjuhaohaoxuexi/p/16412976.html)

[word piece 分词](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter6/6%3Ffw%3Dpt)

[Unigram Language Model 分词](https://link.zhihu.com/?target=https%3A//huggingface.co/course/chapter6/7%3Ffw%3Dpt)

[why padding](https://link.zhihu.com/?target=https%3A//medium.com/%40canerkilinc/padding-for-nlp-7dd8598c916a)

模型优化（optimization）
------------------

我把模型优化放到了模型前面，因为本章将会引入机器学习一个重要的概念——梯度下降，它是机器学习训练的核心，并且会以梯度下降为核心展开模型优化算法内的学习率和损失函数等概念

### 梯度下降

我们知道，机器学习本质是将一个输入的 $M * N_1$M * N_1 矩阵通过计算转化为另一个 $M * N_2$M * N_2 的输出矩阵，而模型内的所有参数都是以矩阵的形式存在的，模型训练即用训练集的输入输出去逐步地优化模型参数，让模型在训练集上的输出更加贴近实际输出（比如线性回归就是一个梯度下降优化的过程）。

本节的文档： [线性回归是什么](https://zhuanlan.zhihu.com/p/72513104)，[手推梯度下降](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/pinard/p/5970503.html)

本节你需要了解：

*   什么是梯度下降
*   梯度下降是如何计算的
*   手推一下梯度下降的公式

### 学习率

上一节我们知道了最简单的 BGD 和 [SGD](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=SGD&zhida_source=entity)，但是在实际场景中我们经常会遇到鞍点问题，而传统的基于固定学习率的梯度下降算法无法有效的解决鞍点，因此在大模型的训练里我们会用到[更多看起来很花哨的梯度下降算法](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/guoyaohua/p/8542554.html)

本节你需要了解：

*   鞍点是什么，如何解决训练中的鞍点问题
*   定量了解 [momentum](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=momentum&zhida_source=entity)、adagrad、adam 优化算法的原理

### 损失函数

模型训练时，我们需要一个函数去评估模型的输出和实际输出的损失有多大（也可以理解为误差有多大）

需要看的文档： [损失函数是什么, 代码里如何编写](https://link.zhihu.com/?target=https%3A//www.geeksforgeeks.org/ml-common-loss-functions/)，[交叉熵损失函数](https://www.zhihu.com/tardis/zm/art/35709485?source_id=1003)，[最大似然估计](https://zhuanlan.zhihu.com/p/26614750)

本节需要了解：

*   损失函数是什么，最好代码层面实现一下
*   交叉熵损失函数和[最大似然函数](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E5%87%BD%E6%95%B0&zhida_source=entity)是什么，解决了什么问题
*   定量了解交叉熵损失函数和最大似然函数的推导
*   [交叉熵函数](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=%E4%BA%A4%E5%8F%89%E7%86%B5%E5%87%BD%E6%95%B0&zhida_source=entity)和最大似然函数的区别是什么

### 正则化

正则化就是避免模型训练中的过拟合现象，通过引入噪音，清洗数据集的方法让模型的训练更不容易出现问题

需要看的文档：[什么是正则化](https://link.zhihu.com/?target=https%3A//0809zheng.github.io/2020/03/03/regularization.html)，[标签平滑](https://link.zhihu.com/?target=https%3A//blog.csdn.net/HUSTHY/article/details/115346629)

本节需要了解：

*   正则化是什么，有什么作用
*   尝试用代码写正则化

### 激活函数

为什么需要在单个神经元的输出经过激活函数：1. 解决线性不可分问题；2. 避免梯度消失和[梯度爆炸](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8&zhida_source=entity)；3. 归一化

具体文档可查看[这篇](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzA3MzI4MjgzMw%3D%3D%26mid%3D2650732724%26idx%3D4%26sn%3D5230b8bb1811cda38ab97afb417d1613%26chksm%3D871b3ccab06cb5dcdf0bdfadcc7ae85d8ae95588bed0b884a55ba50b76d541771104675fbb3e%26scene%3D21%23wechat_redirect)

本节需要了解：

*   激活函数常见的有哪些
*   定量了解为什么需要激活函数

模型
--

### Transformer

**transformer 架构可以说是一切 [llm](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=llm&zhida_source=entity) 的根源**

这里强烈推荐把 [transformer 论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.03762)全篇阅读，第一次阅读能对模型有一个定性地认知即可

读完论文后你肯定有很多不解的地方，可以直接搭配[中文详解](https://zhuanlan.zhihu.com/p/420820453)再精度一遍，对论文的讨论有定量的认知，如果觉得不够还可以看[这个](https://link.zhihu.com/?target=https%3A//medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3)，如何写代码可以参考[这个](https://link.zhihu.com/?target=https%3A//nlp.seas.harvard.edu/2018/04/03/attention.html)，BERT 和 transformer 的区别可以参考[这个回答](https://link.zhihu.com/?target=https%3A//ai.stackexchange.com/questions/23221/how-is-bert-different-from-the-original-transformer-architecture)

然后再读[这个](https://zhuanlan.zhihu.com/p/360144789)了解为什么 [transformer 模型](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=transformer%E6%A8%A1%E5%9E%8B&zhida_source=entity)是这个形态，即回答为什么用论文中的方案能很好的解决 transformer 模型所面对的问题

本节需要了解：

*   transformer 原始论文中用到了什么 embedding？为什么 transformer 模型架构需要引入额外的[位置编码](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81&zhida_source=entity)？论文中如何生成词语位置编码？
*   transformer 论文中如何解决变长输入？
*   transformer 的模型架构是什么样的（能在脑子里立即想出来）
*   什么是 [self attention](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=self+attention&zhida_source=entity)？具体的公式是什么样的？
*   为什么需要将原始输入 $X$X 转化为 $Q,K,V$Q,K,V 三个矩阵计算？直接将 $K$K 和 $Q$Q 使用同一个权重矩阵生成行不行？为什么 [self-attention](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=self-attention&zhida_source=entity) 公式内的 softmax 需要除以 $\sqrt{d_k}$\sqrt{d_k}
*   什么是 multi head attention？为什么需要 multi head attention？为什么 multi head attention 要降维？
*   attention mask 是什么？有什么作用？
*   transformer 论文中的 encoder 和 decoder 有什么区别？
*   以翻译为场景思考下 encoder 层的输出如何进入 decoder 层内进行误差计算
*   为什么 transformer 论文中使用 batch norm 做归一化而不是 layer norm

### BERT

本节需要把 [BERT 论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1810.04805)定性阅读，相比 transformer，BERT 引入了 [pretrain](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=pretrain&zhida_source=entity)+fine tuning 的训练模式，同时也只使用了 transformer 的 encoder 部分，为后续的 llm 训练范式打下了基础

定性读完后可以结合[中文详解](https://zhuanlan.zhihu.com/p/46652512)定量了解 BERT，然后再阅读[这篇](https://zhuanlan.zhihu.com/p/144026536)学习为什么选择用 BERT 和预训练 + 精调来解决 NLP 的问题

本节需要了解：

*   BERT 的模型架构是什么？和 GPT（注意这里可不是 openai 那个 GPT，而是最初的 GPT 模型）有什么不同？和 transformer 比起来有什么不同？
*   BERT 论文中用到了什么 embedding？
*   BERT 论文中的位置编码和 transformer 有什么不同？
*   pretrain 是如何做到的？训练集长什么样？为什么说是[无监督训练](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=%E6%97%A0%E7%9B%91%E7%9D%A3%E8%AE%AD%E7%BB%83&zhida_source=entity)？
*   为什么激活函数从 ReLU 换成了 [GELU](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=GELU&zhida_source=entity)（定性了解即可）
*   fine-tuning 是如何在 pretrain 后的模型上做到的？
*   BERT 在哪些任务上有更好的表现？

其实 BERT 是 transformer 之后在 [nlp 领域](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=nlp%E9%A2%86%E5%9F%9F&zhida_source=entity)被广泛推崇的模型，后来出现了很多基于 bert 的模型（如 ALBERT 等），不过由于原理相似，读者可以自行了解，这里不再展开

### T5

老样子，[T5 论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1910.10683.pdf)先看看，这篇论文其实也没有太多定量的内容，其内容就是大型 [Seq2Seq](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=Seq2Seq&zhida_source=entity) 的 BERT + 干净的数据 + 多任务 + 一些改动的整理，论文的作者深入对比了不同的预训练目标、模型结构、无监督数据集、迁移方法、[NLU](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=NLU&zhida_source=entity) 任务，最终拼成了 T5。你可以简单理解为在 transformer 基础上做了大量的模型参数时延最终得到了一个结论，而基于这个结论搭建的模型就叫 T5

T5 论文将所有 NLP 问题归结为了 text-to-text 任务，并引出了训练数据对训练结果的影响 (论文中的 C4 数据集)，然后通过大量实验得出预训练的目标（如 mask 掉多少，mask 平均长度为多少），最终得到了一系列基于 transformer 的结论，**简单来说就是个实验报告**，详细可以看[中文详解](https://zhuanlan.zhihu.com/p/88363572)

T5 用到的位置编码可以看[这篇论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1803.02155.pdf)

本节需要了解：

*   各个 NLP 领域的问题是如何在 T5 论文中被转化为 text-to-text 任务的？
*   T5 做了哪些实验？得到了哪些结论？
*   论文中的 T5 参数量有多大？
*   T5 用到的 relative position embeddings 是什么原理？

### GPT

**请注意，这里的 GPT 并不是指 openai 的那个 GPT 产品，而是最早的 Generative Pre-Training 模型**。其实当 T5 问世的时候，很多人认为 t5 就是 nlp 领域的最终解，后续要做的无非是优化训练集加大参数量罢了。但是现在我们都知道，gpt 的异军突起直接把 t5 干趴下了，这也是我把 GPT 放到 T5 之后介绍的理由，即使 GPT 模型本身是早于 T5 的

老规矩先定性阅读[论文](https://link.zhihu.com/?target=https%3A//cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)，然后再精度[中文解读](https://zhuanlan.zhihu.com/p/412351920)

GPT 架构选择了 transformer 内的 decoder，因此训练时不会关注后向词语的关注度信息，其 pretrain 的训练目标是极大化词语序列的似然估计，fine-tuning 就是极大化精调层的目标函数，如果你弄懂了 transformer 和 BERT 的话，GPT 原理其实十分简单

本节需要了解：

*   GPT 的模型架构和 BERT 有什么区别？和 transformer 有什么区别？
*   GPT 论文中的模型参数量是多少？
*   GPT 是如何完成预训练和精调的？

### GPT-2

老规矩 [GPT-2 的论文](https://link.zhihu.com/?target=https%3A//cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)先定性读完，然后看看[中文翻译](https://zhuanlan.zhihu.com/p/613698609)

GPT 论文中提到在 zero-shot 的设定下，模型的表现能力与解码器层的数量呈正相关，因此我们看到 GPT-2 相比 GPT 而言，transformer 层翻了 4 倍（48 层），因此参数量也变为了 1.5B(1542M)。同时 GPT-2 的训练用到了自行清洗的 WebText 的数据集（可以看出后续的 LLM 训练对数据集的要求都很高），该数据集从 Reddit 上收集了至少有 3 个赞的外部链接文本，覆盖领域十分广阔，GPT-2 论文中指出规模大的模型必须要用更多的数据才能收敛，最终的实验结果也表明 GPT-2 的模型仍然处于一个欠拟合的情况。架构方面，GPT-2 依然使用 [Decoder](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=Decoder&zhida_source=entity)，并做出了一些细节上的修改（前置层归一化和后置层归一化）。GPT-2 的目标是证明有一种模型完全不需要对下游任务进行适配就可以表现优异，因此论文作者从 WebText 数据集中清洗出各个 NLP 任务的训练数据（问答、翻译等），实验结果也证明 GPT-2 的泛化能力十分强大，在 8/9 个 NLP 任务里都达到了当时的 SOTA

GPT-2 引出了 [zero-shot learning](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Zero-shot_learning) 的概念，可以简单理解为不提供答案上下文的情况下询问 llm 问题，如`把中文翻译成英文： 我 => ？`是 zero-shot，`把中文翻译成英文： 你 => you， 我 => ？`是 one-shot。当然，受限于参数规模，GPT-2 本身在 one-shot 和 few-shot 的表现上也不尽人意

本节需要了解：

*   什么是 zero-shot，什么是 one-shot
*   GPT-2 和 GPT 在模型架构上有什么不同
*   GPT-2 用到了什么分词方法
*   GPT-2 在 zero-shot 和 one-shot 上的表现如何？

### GPT-3

先看 [GPT-3 的论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2005.14165.pdf)，然后[中文翻译版](https://zhuanlan.zhihu.com/p/613693427), 到这里也可以读一下 [GPT 系列的总结](https://zhuanlan.zhihu.com/p/412351920)

GPT-3 延续了 GPT-2 的大力出奇迹思路，直接把模型的参数量提升到了 175B（对比 GPT 的 0.15B 和 GPT-2 的 1.5B），并且继续探索了不对下游任务进行适配时模型的表现（即依然不做任何 fine-tuning）。不同于 GPT-2 的 zero-shot，GPT-3 旨在探索大模型的 In Context Learning 能力，即根据问题上下文进行学习并给出解答的能力，就是我们之前提到的 few-shot，而 GPT-3 的评估也用到了 zero-shot、one-shot 和 few-shot 三种条件进行，结果也显示模型越大，上下文学习学习能力就越强

GPT-3 的预训练方法和 GPT-2 类似，不过 GPT-3 扩大了模型的大小（显而易见的，毕竟从 1.5B 到了 175B）、数据集大小和多样性以及训练文本长度，同时也设置了四种不同上下文的 few-shot 模板来精调，从论文的量化评估来看 GPT-3 也确实做到了对其他模型的降维打击

总结一下，从 GPT 到 GPT-3 我们可以发现，[OpenAI](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=OpenAI&zhida_source=entity) 选择了和 Google 完全不同的道路，当 Google 在深度探索 pretrain+fine-tuning 解决单一场景问题时，OpenAI 则是在大力出奇迹（只用 pretrain 覆盖所有 NLP 场景）的道路上越走越远，直到 GPT-3 的问世终结了这一竞争

本节需要了解：

*   GPT-3 是如何训练的？
*   GPT-3 和 GPT-2 有何不同？

Parameter-Efficient [Fine-Tuning](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=Fine-Tuning&zhida_source=entity) & Prompt Tuning
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

GPT-3 之后，大模型的概念逐渐明朗（也很好理解，毕竟 GPT-3 都有了 175B 的参数量了），但是很明显，此时的显卡并不能很好的装载这些大模型，以 FP16 半精度格式计算，一个 175B 的 GPT-3 模型都需要 320GB 存储空间了，而训练时的梯度数据则更是会成倍的增加显存的占用，这将导致模型的训练时间变得恐怖，以前一天就能训练好的模型现在则需要数月，而且单个任务的全参数微调也容易造成通用性的损失，容易使模型陷入过拟合中

为了解决上述问题，从 GPT-2 开始，一种新的思维被提出：与其全参微调模型，不如固定模型大部分参数，在 Transformer 层中添加额外的参数，并只微调这一小部分来达到 Fine Tuning 的效果，这种范式我们称之为 [PEFT](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=PEFT&zhida_source=entity)（Parameter-Efficient Fine-Tuning），其主要有 Adapter Tuning 和 [LoRA](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=LoRA&zhida_source=entity) 等。不过随着 Prefix Tuning 和 P-Tuning 等被提出，一个更加有意思的命题也诞生了——Prompt Tuning。

不同于 PEFT 在 Transformer 层添加额外参数训练，Prompt Tuning 似乎更加喜欢修改输入序列的 embedding，类似于在把输入语句嵌入一个模板一样。Prompt Tuning 也被分为 Discret Template（可以理解为添加离散的 token 序列）和 Continuous Template（可以理解为生成连续的 embedding），其中比较关键的几种方案有 Soft Prompt Tuning， Prefix Tuning，P-Tuning 和 P-Tuning V2，它们的异同我将在本章最后展开

### Adapter Tuning

![](https://pica.zhimg.com/v2-1fecb0d8c162d53077f46216080e95f0_r.jpg)

adapter tuning 就不用看论文了，其原理一张图就能看懂。精调时，在预训练好的 transformer 模型中的每个前向传播层后添加一个 `[adapter layer](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=adapter+layer&zhida_source=entity)`，该层会对输入进行一次降维和升维操作，并且为了防止最差的情况也设计了`skip-connection`结构（可以直接 identity）

该方法最终用 3.6% 的参数量获得了和精调差距在 0.4% 以内的效果，并且收敛速度也大幅度增加

![](https://pic4.zhimg.com/v2-e872d8c6bf5626e48d1213c13dd14787_r.jpg)

Adapter Tuning 其实和后来的 Prompt Tuning 没有太多关系，它主要证明了对于 Transformer 架构的模型，通过固定模型参数 + 添加部分可训练参数的精调模式是可行的，为后来的 Prompt Tuning 打下了基础

本节需要了解：

*   定性了解 Adapter Tuning 的原理

### LoRA

[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2106.09685.pdf)和[解读](https://zhuanlan.zhihu.com/p/646791309)

LoRA 其实和 Adapter Tuning 类似，是一种低资源量微调大模型的方法，它的原理其实很简单，给 Multi Head Attention 内的 $W_Q, W_K, W_V$W_Q, W_K, W_V 矩阵添加一个低维 0 映射 + MLP + 高维映射来更新参数，这些低秩分解的输出最终会和矩阵输出求和，然后精调时冻结模型只更新这些低秩参数，和 Adapter Tuning 比起来，由于它的参数计算是并行，所以它并不影响模型的推理速度，换句话说，精调速度很快

![](https://pic2.zhimg.com/v2-a7e4013e20b6d899ee80caa7d53bfaf7_r.jpg)

和 Adapter Tuning 一样，这个东西不太需要定量了解，只需要知道它证明了大模型内在秩和小参数量更新是有效的即可，如果你要定量了解原理的话可能你还得去复习下线代里矩阵秩的相关知识

本节需要了解：

*   定性了解 LoRA 的原理

### Pattern-Exploiting Training

PET（Pattern-Exploiting Training）可以简单理解为人工构建模板，即 Discrete Template，这种构建出来的模板称为 Hard Prompt，该方法将情感分类等 NLU 任务通过完型填空的模板变为 [NLG](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=NLG&zhida_source=entity) 任务，它通过在输入中嵌入人工固定的语句来增加大模型在某类任务上的评估指标。举个例子，现在有感情分类任务`I love this movie.`，为了让这个输入更贴近预训练的预料，我们可以将其改为完型填空的格式`I love this movie. The movie is <MASK>`，然后让模型预测下一个词（即这句话的情感分类结果），如此便将 NLU 任务变为了 NLG 任务

这种手动构造 prompt 的模式有一定的局限性，因为离散化寻找出的结果可能不是最优，而且对 token 的变动十分敏感，所以后续的研究大多都是基于 Continuous Prompt 进行，相关的[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2001.07676v3.pdf)感兴趣可以看一看

本节你需要了解：

*   PET 是如何把 NLU 任务转化为 NLG 任务的？

### Soft Prompt Tuning

[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2104.08691)和[代码](https://link.zhihu.com/?target=https%3A//github.com/kipgparker/soft-prompt-tuning)

![](https://pic3.zhimg.com/v2-8dad4b8f915dd7dca5e49d8dc8f0f1dc_r.jpg)

Soft Prompt Tuning 的原理很简单，从清华给出的[代码](https://link.zhihu.com/?target=https%3A//github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py)里就可以理解。在 tuning 阶段冻结模型的参数，然后将输入文本的 embedding 的输入序列左侧拼接一个 soft embedding 作为最终的输入 embedding 传递给 Multi Head Attention 层（注意这里的 Soft Prompt 也是占用 input seq length 一部分的），tuning 阶段的目标便是训练这个 soft embedding，所以这种 PEFT 的方法又称为 Soft Prompt，其原理如下图所示：

![](https://pic1.zhimg.com/v2-c7c3c8666c264689fd08a794981755e8_r.jpg)

Soft Prompt Tuning 是一种 Continuous Template，针对的模型是 T5 等 MLM 的 NLG 任务，但很明显这种 prompt 不具备通用性，每种下游任务只能精调一种，而且作为早期的 Prompt Tuning 方法，和后面的 Tuning 方法比起来，其评测指标已经没有了参考价值

本节需要了解：

*   Prompt Tuning 的原理，看懂代码实现
*   Prompt Tuning 是如何修改 embedding 的

### Prefix Tuning

先附上[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2101.00190)和[中文翻译](https://zhuanlan.zhihu.com/p/639685912)，看完后可以再看看[这个](https://zhuanlan.zhihu.com/p/673985751)里的 prefix-tuning 部分和[实现代码](https://link.zhihu.com/?target=https%3A//github.com/XiangLi1999/PrefixTuning)

![](https://pic3.zhimg.com/v2-fb80b789c0f66892278b712b5e6b38b8_r.jpg)

Prefix Tuning 你可以理解为给每个 Transformer 层的输出都加了不同的 Soft Prefix（注意 Soft Prefix 是不占用 seq length 的），tuning 时只更新这些 Soft Prefix 的参数。不过在具体实现上，为了防止直接更新 Prefix 的参数导致训练不稳定和性能下降，论文作者在 Prefix 层前面添加了一个 MLP 结构（每层都保留一个），通过一个小维度 embedding 和每层的 MLP 生成每层的 Soft Prefix [Embedding](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=Embedding&zhida_source=entity)，增加了可训练的参数量，在训练结束后模型只保留 MLP 生成后的 Prefix Embedding。其具体到某个 Multi Head Attention 层计算的原理如下图所示（注意里面忽略了 softmax 等操作，只关注矩阵的 dim 变化）：

![](https://pic2.zhimg.com/v2-32e3979bee45b0aea6502a1e986299e3_r.jpg)

Prefix Tuning 也是一种 Continuous Template，论文的实验中用在 GPT-2 和 BERT 等模型的 NLG 任务

本节需要了解：

*   Prefix Tuning 中，每层 transformer 层输入添加的 prefix 是如何生成的？
*   论文中 Prefix Tuning 可训练的参数量和模型整体相比如何？
*   prefix 和 prompt 有什么区别？

### P-Tuning

[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2103.10385)和[解读](https://zhuanlan.zhihu.com/p/364141928)，[代码](https://link.zhihu.com/?target=https%3A//github.com/THUDM/P-tuning)

![](https://pic2.zhimg.com/v2-f6a5eebdbda586f77a17939d1330af15_r.jpg)

P-Tuning 其实和 Soft Prompt 相似，通过给输入序列的 Embedding 前添加额外的 Soft Prompt Embedding 实现 tuning，训练时冻结模型参数，只训练 Soft Prompt Embedding 参数。不过论文是通过一个小型的 [LSTM](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=LSTM&zhida_source=entity) 给出 Soft Prompt Embedding，并且作者表示在训练 SuperGLUE 数据时甚至没有冻结预训练模型的参数，这一系列实验也确实证明 Prompt Tuning 这个思路是可行的。

![](https://pica.zhimg.com/v2-62217fd7040527482876357250f8c56c_r.jpg)

P-Tuning 更加聚焦于 NLU 任务（这也是此前 GPT 一直不擅长的领域任务），借助 P-Tuning，GPT 首次在 SuperGLUE 上成绩超过了同等级别的 BERT 模型，颠覆了一直以来 “GPT 不擅长 NLU” 的结论，这也是该论文的题目由来。

本节需要了解：

*   P-Tuning 中有哪些参数被训练的，这些参数是如何注入到模型中的？
*   P-Tuning 和 Prefix Tuning 有什么异同？

### P-Tuning V2

[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2110.07602.pdf)，[解读](https://zhuanlan.zhihu.com/p/673985751)，[代码](https://link.zhihu.com/?target=https%3A//github.com/THUDM/P-tuning-v2)，和 P-Tuning 一个团队写的论文

P-Tuning V2 和 Prefix Tuning 很相似（甚至 Huggingface 上的实现这俩都一样），通过给每个 Transformer 层添加一个 Soft Prefix 来精调，其原理和 Prefix Tuning 基本没有差异，不过论文中做了大量 NLU 任务实验，同时做了一些模型架构和训练上的改进：移除重参数化的编码器（比如 Prefix Tuning 的 MLP 和 P-Tuning 的 LSTM），针对不同任务采用不同 Prefix 长度，引入 Prompt 预训练 + 下游适配，分类任务回归传统分类标签范式

本节需要了解：

*   P-Tuning V2 和 Prefix Tuning 有什么不同？
*   P-Tuning V2 和 P-Tuning 有什么不同？

### ICL

[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2205.05638.pdf)，[解读](https://zhuanlan.zhihu.com/p/609954679)

![](https://pic2.zhimg.com/v2-ad9d21a50bc249973a72f5f99ec70ae9_r.jpg)

ICL 作为 PEFT 方法，在 $K$K ， $V$V 和非线性层后点乘一个可训练参数，精调时只训练这几个参数。不过该论文不止提出了一种 PEFT 方法，也提出了叫做 T-Few 的训练方法，这个可以自行了解了

### 一点小总结

对于上述的 PEFT 范式，这篇高分[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2110.04366)给了一个很好的全局视野，作者在不同的 NLP 领域任务上做实验对比不同 PEFT 方法的性能，并提出了一种 Mix-And-Match Adapter 的方法融合了 Prefix Tuning 和 Scaled Parallel Adapter 结构。我们在精调自己的 LLM 模型时，便可以结合自己的场景参考这篇论文选择合适的精调方式。

[RLHF](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=RLHF&zhida_source=entity)
-------------------------------------------------------------------------------------------------------------------------

RLHF 也是 InstructGPT 精调时主要用到的方法，因此在介绍 Instruction Tuning 前，我们先看看 RLHF 的相关概念

本章引用的部分论文不需要完全阅读，因为后面 Instruction Tuning 会展开介绍

### [Reinforcement Learning](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=Reinforcement+Learning&zhida_source=entity)

先简单介绍一下 RL（Reinforcement Learning）

![](https://pic2.zhimg.com/v2-c60240dcbf3d806cb12a07204191be2f_r.jpg)

如图所示，被训练实体 $Agent$Agent 从环境 $Environment$Environment 收到 $t$t 时刻动作 $a_t$a_t 的反馈 $r_t$r_t 和实体状态 $s_t$s_t 后，对 $t+1$t+1 时刻的环境采取行动 $a_{t+1}$a_{t+1} , $t+1$t+1 时刻的环境再次给予实体反馈 $r_{t+1}$r_{t+1} 和实体的下一个状态 $s_{t+1}$s_{t+1} 。符合这一范式的学习过程都可以称之为 RL，比如马尔科夫决策过程就是一个 RL 过程，[贪心算法](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=%E8%B4%AA%E5%BF%83%E7%AE%97%E6%B3%95&zhida_source=entity)也是一个 RL 过程。对于 RL 我们了解到这个程度即可，深入了解诶的话就涉及到很多和 LLM 无关的数学推导了

### 范式

我们再来看 RLHF，RLHF（Reinforcement Learning from Human Feedback）就是用强化学习的方式依据人类反馈去优化语言模型

RLHF 框架有三步：

*   预训练语言模型：选择自己的模型架构 (Encoder？Decoder？激活函数？归一化层？)，选择自己的预训练数据集，开启无监督训练
*   训练奖励模型 RM（[Reward Model](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=Reward+Model&zhida_source=entity)）：训练一个打分模型，这个模型接受一系列文本并返回一个人类偏好值，这个模型可以用与训练模型精调，也可以从头开始训练，架构自选，不过目前主流都认为 RM 应该和生成模型一样需要类似的文本理解能力
*   强化学习微调：利用 Proximal Policy Optimization 算法（后面一节详细展开）和 RM 对预训练语言模型的输出进行微调

这套框架也是符合 RL 范式的，我们基于训练数据集输入 $Prompt$Prompt ，从原始模型 $\pi_{base}$\pi_{base} 和[精调模型](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=%E7%B2%BE%E8%B0%83%E6%A8%A1%E5%9E%8B&zhida_source=entity) $\pi_{tuned}$\pi_{tuned} 中获得两段输出 $\pi_{base}(y|x)$\pi_{base}(y|x) 和 $\pi_{rl}(y|x)$\pi_{rl}(y|x) ，然后利用奖励模型 $RM$RM 对两个结果进行打分得到分差 $r_\theta(y|x)$r_\theta(y|x) （精调模型评分和原始模型评分分差自然是越大越好），然后用 [PPO 算法](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=PPO%E7%AE%97%E6%B3%95&zhida_source=entity)来更新模型参数

当然整个流程也会有很多细节，比如分差也需要考虑大模型大幅偏离原始模型的问题，避免输出乱码的情况，接下来会对细节做梳理

### Reward Model

![](https://pic3.zhimg.com/v2-724df42c0f824ecb08b56bf99d1f8960_r.jpg)

RM 要干的事说简单点就是文本多分类器，把每一段文本都分类到某一个得分上，RM 的训练是监督训练，训练数据喂给待精调模型，生成的数据人工打分从而得到训练集，其中得分排名的构建可以参考 [Elo](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=Elo&zhida_source=entity) 或者人工构建，[这里](https://link.zhihu.com/?target=https%3A//huggingface.co/datasets/Anthropic/hh-rlhf)有开源的 RM 训练数据可以参考

对于 RM 的模型架构，论文中大多数用预训练好的大模型进行下游精调得到，因为模型也需要有语言理解能力来对文本进行解读。不过从论文中来看，RLHF 系统的 LM 和 RM 未必需要大小相同：

*   OpenAI 使用了 [6B 的 RM](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2203.02155.pdf) 微调 175B 的 GPT-3
*   DeepMind 使用同一个 [70B 的 Chinchilla 模型](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2209.14375.pdf)作为 LM 和 RM

### 精调

![](https://picx.zhimg.com/v2-d2b9bb6c4b4feb7117dc10b83facad27_r.jpg)

在获得了预训练大模型 LM 和 RM 后，接下来便是用训练集和 RM 来对 LM 进行精调

首先我们确定 $Reward$Reward 的计算方法，对于两个模型 $\pi_{base}$\pi_{base} 和 $\pi_{tuned}$\pi_{tuned} ，我们能得到 RM 的打分差值 $r_\theta'=r_{base}-r_{tuned}$r_\theta'=r_{base}-r_{tuned} ，但是为了防止精调模型过于远离原始模型导致语序混乱，OpenAI 和 DeepMind 都给公式添加了词分布序列的 [KL 散度](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=KL%E6%95%A3%E5%BA%A6&zhida_source=entity)（KL 散度可以自行搜索了）作为惩罚，使得最终的 $Reward$Reward 为 $r_\theta=r_\theta'-\lambda r_{KL}$r_\theta=r_\theta'-\lambda r_{KL}

获得了 $r_\theta$r_\theta 后，我们再用 RL 的优化算法针对精调模型进行参数更新 $\theta+\nabla_{\theta}J(\theta)$\theta+\nabla_{\theta}J(\theta) ，不同的 RLHF 论文也会使用不同的优化算法（大多论文用的是 PPO 算法），这个也可以自行阅读论文了解不同算法在不同模型上的差异性，具体算法的原理由于涉及太多太多数学公式推导，这里就不展开了，感兴趣的可以自行在[这里](https://link.zhihu.com/?target=https%3A//hrl.boyuai.com/chapter/2/dqn%25E7%25AE%2597%25E6%25B3%2595)查看

Instruction Tuning
------------------

和 PEFT 以及 Prompt Tuning 不同，Instruction Tuning 的目的是激发语言模型的理解能力而不是补全能力。Instruction Tuning 并不会冻结模型，它针对每个任务都生成 instruction（可以理解为 Hard Prompt），并在若干个 full-shot 任务上进行微调，最后在具体的任务上进行评估 zero-shot 泛化能力

Instruction Tuning 其实主要是训练数据集的范式，原理上都不难理解，我们学习时其实更加需要关注不同方法的评估结果，要知道 [instructGPT](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=instructGPT&zhida_source=entity) 就是靠 Instruction Tuning 训练出来的

### [FLAN](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=FLAN&zhida_source=entity)

[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2109.01652.pdf), [解读](https://zhuanlan.zhihu.com/p/558286175)

这篇论文首次提到 Instruction Tuning 的概念，并提出一种叫 FLAN 的 tuning 方法——一种通过提升语言模型对 instructions 的理解能力从而提高语言模型 zero-shot 学习能力的方法，该方法在大部分场景下对比 GPT-3 175B 的 one-shot 和 few-shot 都有明显的优势。Instruction Tuning 和 Fine Tuning、Prompt Tuning 的区别如下图所示：

![](https://pica.zhimg.com/v2-585e342dc2b7d9d1d59412a0ac36b476_r.jpg)![](https://pic4.zhimg.com/v2-5b63ccc43653f342cc7f12ad3749d54b_r.jpg)

Instruction Tuning 会针对 NLU 和 NLG 的数据集构建任务，每条数据都会人工构建 10 个模板填入形成新的数据集，评估某个任务时，会把属于该任务的所有数据从训练集中剔除并基于 137B LaMDA-PT 训练模型。最终 25 个训练集中的 20 个上，FLAN 都超过了 zero-shot 的 175B GPT-3

本节需要了解：

*   FLAN 的 tuning 方法

### T0

[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2110.08207.pdf), [解读](https://zhuanlan.zhihu.com/p/558286175)

![](https://picx.zhimg.com/v2-6c82bd8038a118e14e1c943d49ef3ebd_r.jpg)

T0 其实理解起来也很简单，相较于 FLAN，T0 使用了一个更小的 11B encoder+decoder T5 模型，针对 171 个多任务数据集创建了 1939 个精致的 prompt，这些 prompt 都[开源了](https://link.zhihu.com/?target=https%3A//github.com/bigscience-workshop/promptsource)，这些数据全部提供给模型训练，有且只训练一个模型以证明多任务学习能提升模型泛化能力。结果来看，T0 模型用 1/160 的参数量在 8/11 个评估任务中超越了 GPT-3

本节需要了解：

*   T0 的 tuning 方法？和 FLAN 有什么不同？

### InstructGPT

[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2203.02155.pdf), [解读](https://zhuanlan.zhihu.com/p/558286175)

该文章就是 InstructGPT 的论文，文章作者提出了一套基于 RLHF 的训练 Instruction Tuning 方法：1. 通过语言学家生成的数据集对待训练模型精调；2. 对多个 prompt 下的模型输出打分得到打分数据集，然后根据打分数据集训练 Reward Model；3. 构建一个新 Prompt，并通过上一步训练的 Reward Model 给待训练的模型进行 RLHF

其中 InstructGPT 的 RLHF 优化算法使用一种叫 [PPO-ptx](https://zhida.zhihu.com/search?content_id=240138000&content_type=Article&match_order=1&q=PPO-ptx&zhida_source=entity) 的新算法，它将预训练模型的梯度混合进 PPO 的梯度（具体的公式可以查看论文中的公式 2），并得到了更好的效果

![](https://pic2.zhimg.com/v2-fb8059eaf13888364b6918f2c4a9ada1_r.jpg)

该论文的数据都是不开源的（OpenAI 一贯风格），论文大量篇幅都集中评估和对比上，本身原理其实很好理解

本节需要了解：

*   InstructGPT 是如何训练的？

到此，你已经完整的了解了 OpenAI 截止 InstructGPT 的相关技术原理

写在结尾
----

想了想自己曾经也是个数学科班出身的算法工程师，咋就毕业从事云和 PAAS 行业了呢？不过云倒是一直和 ai 结合的紧密，写这篇文章的 idea 其实很早就有了，这次刚好借着大模型的热度一口气梳理完了，后续如果自己再因为什么原因忘了这块算法的知识也能看着复习了 hhhhhh

那么，到此为止这篇文章到这里就介绍完了 GPT-3 和 InstructGPT 了，其实还有很多可以单独起一个文章的算法我都没提到（比如 RL 的一系列优化算法），由于这些涉及了太多的数学推导而且篇幅过长，我后续再考虑整理一下，文章内也可能有遗漏和错误的地方，如果有的话也希望各位读者有功夫指正一下~

References
----------

[1] [https://github.com/thunlp/PromptPapers](https://link.zhihu.com/?target=https%3A//github.com/thunlp/PromptPapers)

[2] [https://zhuanlan.zhihu.com/p/558286175](https://zhuanlan.zhihu.com/p/558286175)

[3] [https://nooverfit.com/wp/15-%E5%A2%9E%E5%BC%BA%E5%AD%A6%E4%B9%A0101-%E9%97%AA%E7%94%B5%E5%85%A5%E9%97%A8-reinforcement-learning/](https://link.zhihu.com/?target=https%3A//nooverfit.com/wp/15-%25E5%25A2%259E%25E5%25BC%25BA%25E5%25AD%25A6%25E4%25B9%25A0101-%25E9%2597%25AA%25E7%2594%25B5%25E5%2585%25A5%25E9%2597%25A8-reinforcement-learning/)

[4] [https://huggingface.co/blog/zh/rlhf](https://link.zhihu.com/?target=https%3A//huggingface.co/blog/zh/rlhf)