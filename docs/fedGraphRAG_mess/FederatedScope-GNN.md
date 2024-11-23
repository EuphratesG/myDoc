# FederatedScope-GNN: Towards a Unified, Comprehensive and Efficient Package for Federated Graph Learning
## intro
This fantastic progress benefits from the FL frameworks, e.g., TFF [5] and FATE [40], which save practitioners from
the implementation details and facilitate the transfer from research
prototype to deployed service.  
prior有联邦的框架，但没有FGL的package  
联邦目前大部分用于cv和nlp，上述框架也不适用于graph。  
FGL目前缺乏dedicated framework，导致很多FGL works（ FedSage+, FedGNN, and
GCFL+）只能from scratch（自己搭建网络）且没有testbeds（数据集和和配套工具）  

challenges： 
1. 传统FL只交换同构数据（模型参数）。异构数据交换（用户之间不只传输模型参数，还要传输节点embedding、加密的邻接矩阵、梯度等等），要处理这些异质化数据就会需要很多不同subroutine，图学习后端还可能不一样（例如一个pytorch一个TensorFlow）。传统FL通常是维护一个静态的计算图再把计算图分给各个用户，但这把压力给到了程序员而不是框架。  
因此需要Unified View for Modularized and Flexible Programming  所谓统一视图即统一后端、统一用message处理异构数据和subroutine。
本文通过在event-driven的框架FederatedScope（把交换的信息统一成message解决本地数据异构，再给每个用户定制handler解决用户行为异构）上构建FS-G
**数据的异质化（heterogeneity）是federatedScope尤其关注的，后面的challenge都是FS-G去额外关注的。**
1. Unified and Comprehensive Benchmarks 防止目前由于隐私问题导致FGL数据集缺少每个用户都用自己各自的数据集，以及每个用户的GNN实现都没有统一的问题
2. FL用到GNN很少，因此从业者缺乏超参数先验。有效且自动的模型调优Efficient and Automated Model Tuning
3. Privacy Attacks and Defence. 传统FL也没有用隐私攻击测试FL是否有隐私泄露风险的先例，FGL传异质化数据更需要保证隐私。
   
## related work
1. FL  
   这里说了作为分布式学习的特例，FL的核心研究方向是优化方法(不尽然，应该是传输数据的隐私保护和优化方法)。fedAvg等论文多用于cv和nlp领域，有survey为证  
2. FGL  
   之前综述里说的FL with structured data中的问题的解决方法基本都要在用户之间传递异构数据（不只模型参数）
3. FL software（framework）
   所谓拆分的computational graph是传统FL框架用来表征各个用户关系和数据流向的，和图结构数据没有关系。用户通常必须用声明式编程（即描述计算图）来实现他们的FL算法，这为开发人员提高了标准。传统FL如此，FGL更不用说了。有一个例外是FedML框架，对应package叫FedGraphNN（不是之前看的FedGraph），但是他也没有考虑上述的3、4问题。

## infrastructure
FS-G based on a event-driven FL framework named **FederatedScope**, which abstracts the data exchange in an FL procedure as **message passing**
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@main/202311031049639.png)
得益于**FederatedScope**，我们可以把一整个FGL procedure的异构数据->messages subroutine->handlers。这样可以不用再static的computational graph中去协调participants。可以十分方便地在FS-G框架下实践任何FGL方法，只需要好好定义message和handler就够了。    

## graphDataZoo
Dataset, Splitter, Transform（将每一个图映射成新图例如将节点的度作为新的节点特征）, and Dataloader（遍历图或者从图中采样子图的集合）我们重点说前两者。  
splitter：通过现有standalone datasets去模拟FGL去中心化的数据集，做的就是把已有的完备大数据集分给各个client。node/link level每个client拥有子图，graph level每个client拥有所有图集合的子集。细节在附录A  
datasets：FS-G三个数据集
## GNNMODELZOO AND MODEL-TUNING COMPONENT
GNNModelZoo类似pytorch作为框架的部分，提供了图神经网络的搭建功能（encoder、GNN、decoder、readout）。尤其nn框架中还包含GNNs that decouples feature transformation and propagation（将特征提取和信息传播分开，解开对网络深度和性能的限制例如 GPR-GNN）等诸多现成的GNN网络。ModelZoo和nn是一体的。    
MODEL-TUNING COMPONENT：  
why？因为调参本身就是重复试错的过程，而FGL的client之间存在大量信息的exchange，因此一个训练的course会开销更大，更经不起反复的调参。  
multi-fidelity HPO 多精度超参数优化，通常存在多个不同成本和精度的评估方法或评估器。
FS-G allows users to reduce the fidelity（降低成本） by：
1. 每次trial只做有限次FL rounds而不是一整个FL course 
2. 每个round采样远小于N的K个用户

以SHA为例子，Model-tuning Component模块中每个超参数配置都有一个config，每个config都可以扔到FGL runner里面从上次的checkpoint开始跑几个round再保存并获得本config的performance，回到Model-tuning Component后排序。SHA会只保留上半部分表现好的config。  
得益于FS-G对于FGL runner接口的设计和能够save and start FGL course，许多one-shot HPO（单次调优方法）都可以泛化到FGL领域上来。  
另外由于FGL runner的返回值是可配置的，因此从系统角度调优（41）也被FS-G支持。  

monitoring and personalization  
在FGL语境下我们不仅考虑client-side的metrics（本地loss函数）还在server端进行考虑。  
然后引入一堆metrics，The larger these metrics
are, the more different client-wise graphs are。同时还可以被可视化和log（第三方WandB and TensorBoard）  
同时FS-G还提供了可以记录update/aggregate过程中产生的任何数量（quantity）的API，用户可以monitor他们并个性化地tune GNN。  


## experiments
为长久以来缺失的FGL领域的benchmark做一个设置，同时在金融领域场景进行一些实验。分别在三个settings下应用四种GNN网络对三个数据集做测试。  
node-level：FGL的测试使用全局评估（用全图测试数据测试而非子图）
原来所指的cited network指的是其他的学术文献数据集啊。。cora、citeseer、pubmed都是数据集，和FS-G自己构建的FedDBLP作对比  
random-splitter下global表现较好，community_splitter下FGL表现比在全图上训练的要好（某种划分trick导致在GNN上表现变好）  





# FederatedScope
FL目前的数据异质化可以分为如下几种：
1. local data的异质化。Personalized FL（Alysa Ziying Tan, Han Yu, Lizhen Cui, and Qiang Yang. 2021. Towards personalized federated learning. IEEE Transactions on Neural Networks and Learning
Systems PP (2021).）对local training定制化更加剧了数据本身因为产生组织不同和产生设备不同导致的异质化（这里还没有说到图数据FGL的异质化）
2. participants resources的异质化，可能用户的本地资源有差异会出现等的情况。
3. participants behavior异质化。FGL里提到的异质化属于这种，另外不同的backends也属于这种，需要额外对数据做处理。
4. training goal的异质化。这种比较正常，不同corp的目标不一样，fedavg就可以处理这种问题。
