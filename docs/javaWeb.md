# 谷粒商城
单体太臃肿，因此把大应用拆分成小服务功能，每个服务占用一个线程，通信http通信。  
何为分布式系统？分布式的计算机为用户完成一个完整的服务。所谓集群则是某个节点由多台服务器集中一起实现同一业务  
服务之间远程调用常用http+json  
负载均衡 A服务要调用B服务，很多服务器可以实现B服务，要注意负载均衡  
服务注册、发现、注册中心（使得服务之间可以互相感知可用状态防止调用不可用的情况）  
配置中心 统一管理各个微服务的参数状态并且全部如一一改全改  
服务熔断 由于微服务之间通过网络通信，因此如果一个服务超时则容易导致请求积压导致整个业务雪崩。因此引入熔断（多次故障就直接返回默认数据）和服务降级（非核心业务直接快速返回，例如不查数据库了等省时间）  
API网关（API gateway）因为本项目是前后端分离的，前后端之间通过http通信，网关负责把关工作  
微服务都是用sb写的微服务  
虚拟机的端口转发（就是端口映射）每装好一个虚拟机软件都要配置一下和windows的端口对应非常麻烦  
节点ifconfig显示的11.11.11.10是内网ip，而整个集群的外网ip是222.20.94.68  

redis就是一个运行在内存中的高速数据库，类似cache的作用。操作耗时短。  
  
linux环境 docker mysql redis  
windows idea java1.8 mvn  
idea开发后台微服务项目  
vscode开发前端的后台管理系统  


idea需要gradle的项目直接下载好gradle之后放在wrapper。properties下面（压缩包就行）就可以了  


逆向工程搞出基本的增删改查代码简化开发

数据太多不设置外键，因为要省去检查外键的操作  
