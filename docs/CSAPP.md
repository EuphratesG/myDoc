# CSAPP
从某种意义上来说，本书的目的就是要帮助你了解当你在系统上执行 hello 程序时，系统发生了什么以及为什么会这样。  
shell是一种命令行解释器（程序）  
内核是操作系统（程序）总是驻留在存储器中的部分  
指针本质仍是指针变量的值传递，子函数栈中存的是指针变量的值传递，指针变量的地址不会改变。引用则是别名，和变量声明一致（按规则取一块地址叫做x），子函数栈中存的是函数调用时实参的地址且无法改变。  
**服务器的硬件**好理解，其实就是一台性能、稳定性、扩展性等等比我们普通个人PC强的一台机器而已，它也需要搭载操作系统，比如有专门的Windows Server或者各种Linux发行版系统。Web服务器、HTTP服务器、应用服务器、Tomcat、Apache、Nginx……等等的概念。通常来讲，只要运行在服务器系统之上，绑定了服务器IP地址并且在**某一个端口**监听用户请求并提供服务的软件都可以叫**服务器软件**。  
我们旋风式的系统漫游到此就结束了。从这次讨论中要得出一个很重要的观点，那就是系统不仅仅只是硬件。计算机系统是互相交织的硬件和系统软件的集合体，它们必须共同协作以达到运行应用程序的最终目的。本书的余下部分将对这个论点进行展开。  
操作系统内核是应用程序和硬件之间的媒介。它提供三个基本的抽象概念：文件是对/O设备的
抽象概念：虚拟存储器是对主存和磁盘的抽象概念：进程是处理器、主存和/O设备的抽象概念。
最后，网络提供了计算机系统之间通信的手段。从某个系统的角度来看，网络就是一种/O设备。

## 第二章 信息的表示与处理   
c语言指针的两个方面：值（某个存储块/字第一个字节的虚拟地址）和类型（由c编译器维护，机器语言层面不存在）  
### 2.1信息存储  
十进制和十六进制的转换，转10*，转hex/  
一般意义上的字长/32位机，指明整数和指针数据的标称大小，也代表了虚拟地址空间的大小4GB。  
小端格式指最低有效字节在地址数大的那边。网络编程时小大端机器之间发送数据若不处理数据则会发生字节反序，12章处理。  
二进制代码是不兼容的，二进制代码很少能在不同机器和操作系统组合之间移植。  
位向量一个很有用的应用就是表示有限集合，位向量a=[01101001]表示集合A={0,3,5,6}。布尔运算和逻辑运算还真是两个东西，但基本一致只不过真值换成01。  
c语言并没有规定有符号数是逻辑右移还是算术右移，但一般默认算术右移，无符号数必须逻辑右移。java则存在运算符的区分。 
### 2.2整数表示    
#### 无符号数的编码  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/CSAPP1.png)  
无符号数的编码具有唯一性，上述函数是个双射。  
这里的编码指位向量到十进制整数的映射。    
#### 补码编码  
为什么同一个位模式补码解释和无符号数解释的绝对值和是2^w，因为补码表示取相反数的话等价于正的最大权减后面的值，后面的值正负抵消，故最后进一位。引出补码和无符号数的互相转换。  
#### c语言中的有符号数和无符号数  
补码位扩展为什么可以保持大小不变，正数显然，负数每往左扩展一位作差之后真值相当于没变。  





我们已经看到了许多无符号运算的细微特性，尤其是有符号数到无符号数的隐式转
换，会导致错误或者漏洞的方式。避免这类错误的一种方法就是绝不使用无符号数。实际
上，除了C以外很少有语言支持无符号整数。很明显，这些语言的设计者认为它们带来的
麻烦要比益处多得多。比如，Java只支持有符号整数，并且要求以补码运算来实现。正常
的右移运算符>被定义为执行算术右移。特殊的运算符>>被指定为执行逻辑右移。
当我们想要把字仅仅看做是位的集合而没有任何数字意义时，无符号数值是非常有用
的。例如，往一个字中放人描述各种布尔条件的标记(flg)时，就是这样。地址自然地就
是无符号的，所以系统程序员发现无符号类型是很有帮助的。当实现模运算和多精度运算
的数学包时，数字是由字的数组来表示的，无符号值也会非常有用。  

### 2.3整数运算  
无符号加法 溢出就按当前位数取模
、补码加法、  
补码的逆 所谓加法的逆就是相加得0就是相反数 为什么补码连带符号位（其实本质上没有这个概念）取反加一就是补码的逆，因为取反加一就是取模，补码就是基于取模的定义只不过正0和负0摘出去了其他数都是取模的一对 


多核cpu的发展史  
人们对于计算速度的追求是无止境的，单核cpu主频已达到瓶颈（主频代表指令执行频率）  
Simultaneous Multithreading/hyper-threading 超线程技术  
超线程既属于并行，也属于并发。在没有smt时期，单核处理器想切换线程是很麻烦的，需要触发中断、保护现场和写回内存。有了smt后，一个核心可以同时持有两套线程，并且在流水线空缺时执行双线程，这种情况是并行的；如果流水线满利用率，那么smt就是并发的，但这也比只持有单个线程的核心更方便切换。  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/csapp2.png)  
图中四核CPU每个核都有T0、T1作为逻辑多核。  
每个核都有自己的L1/L2/L3 Cache（我们也会看到LLC，Last Level Cache的说法，在这个图里面就是指L3）。根据程序运行的局部性原理，可以猜测并预先从内存里面取一些指令/数据出来。如果是指令，就放入L1 I-Cache（Instruction）；如果是数据，就放入L1 D-Cache（Data）。指令和数据分开的原因是指令的复用程度一般远远高于数据。但是L2和L3为什么不分开呢？这是成本/复杂度/性能之间的权衡。  
多核带来了一致性问题，大大提高了Cache复杂度，也是软件工程师关注的核心问题。  
多核提高性能的思路也是如此。首先CPU操作数据的基本单位是CacheLine，那么按道理只要不是两个人同时操作一个CacheLine，就可以使用不同的锁了，自然并发度就高了，即要为每个CacheLine准备一把锁。进一步，如果某个CacheLine只有一个人写，剩下的人都是只读，那么就没必要加互斥锁，加读写锁会更快。对于任何操作对象，状态越多就越容易高并发，只要状态管理的开销低于并发的好处，整体性能就是收益的。  


预处理（include换成真文本，仍然是文本） 编译（狭义编译，成汇编语言） 汇编（assemble，成机器码） 链接（把预处理的那些链接起来，再加上库函数一起链接得到可执行文件linux和windows不一样，在这一步执行的是静态链接）   
而java则先编译成.class文件


1. 浏览器是最经常使用到的一种客户端程序。
2. 浏览器是一个软件。具体定义：浏览器是指可以显示网页服务器或者文件系统的HTML文件（标准通用标记语言的一个应用）内容，并让用户与这些文件交互的一种软件。

IPC （Inter Process Communication）进程通信

所谓的“连接”，其实是客户端和服务端保存的一份关于对方的信息，如ip地址、端口号等。  
tcp三握手四挥手在互相的视角都要知道两方都准备好了再完成，主要考虑报文丢失

1. 为什么要三握手，二握手不行吗 关键在连接确定决定权要在客户端
   1. 防止旧的连接先于新的连接初始化造成混乱。会导致服务端白白浪费了建立连接的资源和收到RST之前发送的无效数据。
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/browser1.png)
   1. 同步双方初始序列号syn 四次握手其实也能够可靠的同步双方的初始化序号，但由于四握手就变成两个半双工连接了，且第二步和第三步可以优化成一步，所以就成了「三次握手」。  
   而两次握手则只能保证一边而两次握手只保证了一方的初始序列号能被对方成功接收，没办法保证双方的初始序列号都能被确认接收。
   2. 防止客户端连接请求超时重传后旧请求延迟到达产生无效连接浪费资源（有点类似1，只不过3是同一个连接请求的重传，1是快速产生新连接）
1. 为什么要四挥手 fin
2. 四挥手结束客户端为什么要等2msl 若第四挥丢失则要等第三挥再来
3. https tcp->ssl(secure socket layer)->https 共享密钥加密和公开密钥加密的混合机制


get和post主要区别就是post可以向服务器提交大数据文件，如上传图片。get只能URL上带点请求参数。post 除了get的功能还能顺便给服务器邮寄（post）包裹。get一般来说是幂等（不对服务端产生副作用），post不幂等，get只用于简单请求url带一点请求参数，post参数一般都在body，但实际可以灵活处理。现状是post几乎完全代替get。

并行和并发要点：串行还是并发，这都是任务安排者视角看到的东西，是为了解决一个问题安排了一些进程串行或并发。前者要求你看到前一个任务结束了，下一个任务才能安排；而后者呢，你可以同时提交许多任务，执行者（们）之间会相互协调并自己安排执行顺序（但未必合理，比如可能出现死锁）。相比之下，“并行”是任务执行者视角的东西，和前两者所处平面不同。另外，同步和异步也是任务执行者（进程）视角的东西。  
换句话说，“并发”的确经常能让“并行”自然而然的出现，硬盘利用率也的确被提高了；只是这种提高缺乏保证（比如，运气不好时，复制进程A可能和进程B争着读取旧硬盘，从而导致很多不必要的寻道动作）；而且，由于并发并不保证合理的执行顺序，反而经常“搬起石头砸自己的脚”，速度不如串行（串行进程处理一个任务一定速度最快，但是资源利用率会很低），因此需要确定进程间的同步关系。 

### 协程
本质上就是用户级线程（工业界一般把进程看做资源分配的单位，线程才是调度单位），而用户级线程本质上是对内核级线程的进一步功能划分，产生目的是减少线程建立、调度、销毁等操作的开销。虽然线程池可以一定程度缓解，但是仍有调度开销。多对一、一对一、多（多）对多（少）。
那就是说，协程A/B/C就是在用户态实现的小型调度器。由用户态的进程/线程代码决定在何时执行协程A/B/C。能理解为协程这个就是用户态状态机吗？比如协程A就是状态a，协程B就是状态b，当协程A发生IO时就切到协程B去执行

所谓回调函数就是中断处理函数，回调属于中断的一种结果。  
进程同步、进程异步  
网络编程 函数的调用方式：同步调用（sync）、异步调用（async）、阻塞调用、非阻塞调用  

1. 同步阻塞（调用者所在线程阻塞且保持轮询）
2. 同步非阻塞（调用者所在线程继续执行且保持轮询）
3. 异步阻塞（这个看起来和同步阻塞很像，但可以这样理解，同步阻塞相当于调用者A调用了一个函数F，F是在调用者A所在的线程中完成的，而异步阻塞相当于调用者A发出对F的调用，然后A所在线程挂起，而实际F是在另一个线程中完成，然后另一个线程通知给A所在的线程，更准确的是将两个线程分别换成用户进程和内核）
4. 异步非阻塞（感觉正常理解意义上的异步）

代理服务器数据没有加密，VPN虽然实际逻辑结构相同但是数据是加密的。  

什么是分布式系统？多个互通的计算机服务器，主要用于分布式存储和分布式计算，将存储或计算进行拆分提高性能，同时解决单服务器的高并发问题。  
问题：
1. 网络通信 网络分区脑裂（部分节点间时延不断增大，最后只有部分节点仍起作用）
2. 三态 成功失败超时
3. 分布式事务：ACID(原子性、一致性、隔离性、持久性)
4. 集群与分布式区别
集群：复制模式，每台机器做一样的事。
分布式：两台机器分工合作，每台机器做的不一样。
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/distributed1.png)   