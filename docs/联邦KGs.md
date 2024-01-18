# 联邦KGs
在分布式机器学习框架中，“计算图”是一个数据结构，它描绘了如何计算每个变量。它由一系列的节点和边组成，其中节点代表数学操作，而边代表数据（如张量）在节点间的流动。在分布式设置中，计算图会被分割，以便不同的参与者可以并行处理各个部分，从而提高计算效率和规模。这种设计允许机器学习模型利用多台机器的计算资源，通过网络分布式地训练。

在传统的机器学习任务中，有两种主要的学习设置："inductive learning"（归纳学习）和 "transductive learning"（转导学习）。例子分别是监督学习和自监督学习。  

注意深度学习中特征向量是按行向量主体，但矩阵运算中还是按列向量主体  
# 戴语言模型
"LLM"通常是指"Large Language Model"，即大型语言模型。大型语言模型是由数十亿个参数构成的深度学习模型，它们能够理解和生成自然语言文本。这些模型通过在大规模文本数据上进行训练，学会语言的统计规律，使它们能够进行文本预测、翻译、摘要、问题回答等多种语言任务。OpenAI的GPT系列（Generative Pre-trained Transformer）就是大型语言模型的典型代表。 

疑问：
1. 大语言模型在文本数据表现很好，但在表格形数据和graph上表现欠佳？肯定啊  
2. 大语言模型的应用两个，用llm来增强已有网络or appstore式把llm作为一个接口？qs
3. 联邦学习server端使用llm作为接口搞一些事情。大模型训练搞不了需要太多资源，那我们就把llm当成一个黑盒。


所谓语义信息应该就是输入向量字面信息分量值之外的信息，例如向量间关系等等。例如知识图谱集成：将结构化知识（如知识图谱中的实体和它们的关系）集成到深度学习模型中，以增强模型的语义理解能力。
GNN可以直接应用于知识图谱上。在知识图谱中，GNN可以用来推断节点的属性（如实体的分类）、预测缺失的关系（链接预测），或是生成知识图谱中不存在的新信息。  


另外知识图谱这个东西和存数据的表格数据库其实本质没有什么不同。只是图结构更好表示关联关系。  
知识有两种存在方式，形式化（知识图谱、数据库）和参数化（模型参数存储知识）  
总的来说，知识图谱是一种特殊类型的图（basic building blocks是三元组），它专注于语义信息和实体间的关系，而不仅仅是数据的结构和连接方式。  

  
LLM用于知识提取任务  
In this article, we provide a comprehensive survey on the evolution of various types of knowledge graphs (i.e.,
**static KGs, dynamic KGs, temporal KGs, and event KGs**) and techniques for **knowledge extraction and reasoning nd augmentation**.  

![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@main/202311131939606.png)  




encoder的输出是feature map，接入softmax就是文本分类。    
decoder其实没有输入，图上写的output是把之前时刻输出的结果当做输入，第一次自己去start初始化  





在使用 Transformer 进行自然语言处理任务时，通常需要一个嵌入层，用于将输入的词语转换为向量表示。这时，可以使用 Word2Vec 学到的词向量作为初始嵌入，以提供一些先验的语义信息。  

编码器-解码器（encoder-decoder）架构是一种常见的神经网络结构，特别适用于序列到序列（seq2seq）任务，其中输入序列被编码成中间表示，然后解码器将这个中间表示解码为输出序列。



 
端到端暂且理解成一气通贯，例如卷积神经网络的训练就不是端到端，因为特征提取和分类是两个训练阶段。  
autoregressive就是当前输出依赖于之前的输出  
encoder-only类型的更擅长做分类；encoder-decoder类型的擅长输出强烈依赖输入的，比如翻译和文本总结，而其他类型的就用decoder-only，如各种Q&A。当然也不绝对，因为三者都是sq2sq的，而且encoder和decoder就差在decoder拥有masked的位置编码而已。例子场景小学生学语文  

zero-shot指测试集中的数据在训练过程中从没用过，模型却能够给出合适的输出。  

1. KG可以在inference and interpret ability上enhance LLM。
2. KG很难evolve by nature，因此无法生成new facts和表示未知的知识。LLM-augmented KGs   
但是，KG和LLM又是紧密联系的。例如一个人口若悬河，说起话来引经据典，那么这个人首先是一个很好的LLM，而且他一定拥有强大的知识储备，即这个人脑子里有一个庞大的KG。这就意味着一个超级的AI系统一定是LLM和KG结合的——KG存储知识，LLM负责语言层面的理解和表达。

未来 多模态LLM强化知识图谱？  


LLM增强知识图谱的一些应用领域：  
KG embedding,KG completion, KG construction，KG-to-text generation,and KG question answering 
传统KG通常是incomplete的，并且尝尝忽视文本化信息  

# KG embedding 把实体和关系都抽象成向量  
LLM编码KG实体和关系的textual descriptions（文本描述）并作为权重加入实体和关系的表示中。  
传统KG很难解决表示unseen实体和long-tailed relations（父子关系和具体 解决方式是加权等等） 所以用LLM搞出新的实体and关系转化为embedding丰富样本。  
Instead of using KGE model to consider graph structure, another line of methods directly employs LLMs to incorporate both the graph structure and textual information into the embedding space simultaneously.
第一种就用LLM embed三元组各自的文本描述特征（text文本描述是从知识图谱里取出后再扩充的，把这些text丢到LLM里面构成一个特征空间），再通过MLP原本的embed方法把这些信息map到三元组各自的embedding中。疑问，不能直接用LLM生成的吗？  
第二种。把三元组中的实体和关系直接也当做文本描述放在text里面（？意义不明）。类似预训练，在已有LLM基础上做针对训练。The LLM is optimized to maximize the probability of the correct entity t. After training, the corresponding token representations in LLMs are used as embeddings for entities and relations.  

# KG completion 也是加入文本特征信息，缺失包括三元组整体的缺失和某个三元组内部的缺失？  
 
LLM输出的embedding的优化  


1. LLM as encoders 这里应该是默认有一个训练好的KG预测MLP，相当于训练不同的模型去实现任务
   joint encoding encode a full triple, 然后预测是不是这个实体。（知道一个完整三元组看看是不是KG里的）
   MLM 知二推三（知道2推剩下一个）
   21分开（知道分开的一个三元组，score function给出拼起来之后在KG里的匹配程度）
2. LLM as decoders
   直接生成t的text，是一句话。  

对比：  
LLMs as Encoders (PaE) **applies an additional prediction head** on the top of the representation encoded by LLMs.因此fine tune的时候可以只操作MLP而不动LLM（？、really？）然而inference阶段要对所有三元组产生一个打分，这是非常expensive的。而且PAE也没法泛化到未知的三元组。另外PaE需要LLM的输出特征，很多SOTA的LLM对此并不开源。  
而PaG则完全把LLM当做黑盒使用，更有前景。当然也存在问题
1. 如生成的实体could be diverse and 不在KG之中
2. 如何设计一个合适的prompt that feeds KGs into LLMs
3. 生成式模型为了self-regression导致的效率问题


BERT是按照使用场景来区分encoder和decoder的，模型的底层结构还是一样的。就是transformer encoder的结构堆叠。  
真正让BERT表现出色的应该是基于MLM和NSP这两种任务的预训练过程，使得训练得到的模型具有强大的表征能力。
MLM、NSP通过大量数据让模型学会预测空白处是什么  

FedE 联邦获得新的实体embedding，然后用于completion。  
差分隐私那篇也只是在联邦表示学习上加了差分隐私、GAN，但是是跨领域的联邦，后期可以细看  
联邦更像是存在一个隐私保护的前提？不同用户有不完整的kg，但是还需要别的用户的kg去补全自己的，但是还涉及隐私保护问题。  
基本都是联邦的embedding，然后去用于**kg补全**。从头开始构建没找到先例  
Heterogeneous Federated Knowledge Graph Embedding Learning and Unlearning 2023  


至于说KG embedding和KG completion的关系？训练过程就是补全，用训练好的embedding去做补全更得劲啊。。  
