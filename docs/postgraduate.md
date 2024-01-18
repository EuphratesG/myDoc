# postgraduate
说明这篇论文的主要贡献、方法特色与主要内容。最慢硕二上学期必须要学会只看Abstract和Introduction便可以判断出这篇论文的重点和你的研究有没有直接关联，从而决定要不要把它给读完。

高性能时序图计算硬件加速机制研究  
科研人员提出了一系列图计算系统或图计算加速器，通过高性能计算、并行计算等技术来优化图计算过程  
复杂的图算法（例如动态图处理、图挖掘、和图学习）不断涌现  

什么是分布式系统？多个互通的计算机服务器，主要用于分布式存储和分布式计算，将存储或计算进行拆分提高性能，同时解决单服务器的高并发问题。  
问题：
1. 网络通信 网络分区脑裂（部分节点间时延不断增大，最后只有部分节点仍起作用）
2. 三态 成功失败超时
3. 分布式事务：ACID(原子性、一致性、隔离性、持久性)
4. 集群与分布式区别
集群：复制模式，每台机器做一样的事。
分布式：两台机器分工合作，每台机器做的不一样。
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/distributed1.png)   


KDD SIGMOD VLDB WWW AAAI IJCAI ICDE这些顶会上每年都有很多关于图或者网络的文章。  

GraphTune: An Efficient Dependency-Aware Substrate to Alleviate Irregularity in Concurrent Graph Processing  23/7/19 ccfa  ACM Transactions on Architecture and Code Optimization  
并发（concurrency）和并行（parallellism）
Concurrent iterative Graph Processing (CGP) jobs可能会用不同的算法去处理同一个图，导致irregularity的产生。  
(1) Irregular memory access: 多个分布式节点虽然操作的图数据不完全相同，但是大部分相同。这导致同样的数据会反复在内存和LLC之间流动，效率低。  
(2) Irregular communication: 多个分布式节点独立发起communication，会产生更多cost，造成网络带宽利用不充分。  

针对这种不规则访问本文发现可以通过 将图的顶点和边按拓扑结构处理去使其规范化。  
顶点状态本质上是沿着图的内在拓扑结构传播的，因此想要访问一个顶点，则肯定需要先访问其周围的顶点去激活它。  
runtime system, called GraphTune 自己命名  
作用是解决不同分布式节点对相同图数据的不规则访问，可能是通过共享来完成这一点。  
By such means, the common chunks can be processed by the related jobs at the same time, and the accesses to them can be fully shared by these jobs.   
GraphTune构建chunk 的依赖关系图以寻求共享，同时设计communication scheme去减少通信开销。  
实际上本文提出的是一个运行时系统，integrate GraphTune with other syss  

