# SICP

书名是Structure and Interpretation of Computer Program。这里Computer说的不是x86机器，而是图灵弄出来的抽象演算机器，或者那种人工计算员。Program不是二进制串，而是那种做规划的，类似算法的东西。  
那么问题就清楚了，这讨论的是，数学上的抽象计算机，命令它干些事情（比如弄点lambda给它做规约，以此来算东西）。这些基本命令，怎么去构造一些高级结构Structure，构造完了，怎么Interpret，也就是怎么规约计算。  
所谓结构structure，就是程序流的构造，和数据结构的构造。怎么弄点更高级的流程控制机制，还有更高级的数据表示机制。后面翻译interpretation，就是讨论怎么把前面构造出来的高级东西描述的“程序”，去计算出来。



函数式编程
procedure的结果还是procedure，这似乎模糊了data和procedure的界限

BLACK-BOX ABSTKACTION   
CONVENTIONN INIERIACES  
基本  
大规模结构  
	面向对象  
	流   
METALLIGUISTC ARSTRACTION  
make new language in terms of  
-》  
primitive elements data and procedures  
means of combination  括号  
means of abstraction  define（what define does is give a name a value）  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP1.png)
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP2.png)  
上面是下面的语法糖 syntactic sugar  
精髓是程序员不需要知道各个procedure是不是原子的procedure，它们只是抽象而已  

cond and if 互相成语法糖都可以  

recursive definition  


## 1B procedures and processes：substitution model：  
至少目前这个交换模型是课程里的machine works的方式  
回去理解第一页ppt  

iteration and recursion  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP3.png)
两种sum计算方式，都是递归定义的程序却在交换模型下产生线性循环、线性递归两个进程。因此语言实现形式并不能决定你写的是循环程序还是递归程序。  
lisp语言没有循环，只能靠递归定义if实现循环。循环和递归同样是有一个判断做出口，同样有变化的参数到下一个状态。但是循环只是以下一个状态做一样的事直到参数达到出口，而递归则是逐渐拆分问题reduce参数，直到达到出口得出结果再加回去，参数变化到出口的次数是递归层数。递归所拆分出的子问题也要做一样的事，只不过可以拆成123任意个状态。  
时间复杂度和空间复杂度 循环O(x)O(1) 递归O(x)O(x) 空间抽象层级单位是一个数字
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP4.png)

斐波那契数列
0 1 1 2 3 5 8 13 21 34 55  
这个例子体现了递归最直接的缺点：会重复计算很多中间结果。例如fib(4)会计算两次fib(2)。  
时间复杂度O(fib(n))空间复杂度O(n)  
为什么人们说编程难，一个原因是你要写出所有instance都fit的程序，例如递归的出口。 
the towers of hanoi  
出口的done和出口是计算的值实际上一样，拆分中间还加其他操作。  


## 2A higher-order procedures  
递归给了我们分解问题成子问题的思想。例如sigma累加也可以写成递归程序。 
-》
procedural arguments 
递归累加的范式程序（procedure作为参数）  
procedure which produces a procedure as its value  
lambda 表达式形式来表示一个匿名procedure，主要是为了简化定义，类似java匿名内部类  

使用牛顿method去求sqrt of x （牛顿法求平方根），牛顿method内部又用了fixed point算法去表达 
牛顿method（作为一个procedure）的作用是找到一个y使得f（y）=0，由一个guess开始找  


一门语言将procedure作为first-class citizens的意义，数据结构还没有涉及
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP5.png)


讨论一个计算机（图灵机）到底能做什么事情，计算机“语言”到底有多少表达能力，这个能力的边界在哪里，我想这才是这本书要回答的问题  
这问题属于谜底就在谜面上，《计算机程序的构造和解释》讨论的核心问题就是计算机程序的构造（写程序）和解释（给程序赋予语义）。 诚如其它答主所说，这本书举了很多奇怪的一般人遇不到的例子，这些例子就是告诉你，这也算计算，也可以构造出这些计算的程序，这些构造出的程序也能解释（运行）出对应的含义（结果）。 前三章用lisp写了一些普通，但又不普通的例子。这些例子告诉了你抽象和lisp的强大，但所有的成功终究是建立再lisp上的，抽掉lisp，这些例子再强大也是镜花水月。后两章则是告诉你如何打下lisp的地基，即如何构造出lisp的程序并解释他，当用lisp解释它的时候，这个程序就是一个普通的解释器，当用硬件解释它的时候，这就是一块可以跑lisp的芯片。 读完全书，你能真正感受到抽象的力量，那一刻，你就仿若置身于整个宇宙，洞察了群星的秘密。之前连那么困难的问题都解决了，这世上应该没有解决不了的问题了吧（这是错觉）。 这种混合着sicp哲学思想的强烈的尤里卡时刻，也是这么多人给予其神作评价的原因吧。


## 2B compound data  
通过有理数加减乘除的例子引出cons、car、cdr  
一个重要理念：为什么要构建data abstraction？例如明明可以直接把pair的底层暴露出来不构建rational number类和numerator、denominator方法（即一个新类的selector and constructor）。  
程序员在写程序的时候不要直接决定所有细节，发挥wishful thinking的理念忽略一些细节，到需要确定细节的时候再去确定。  
let和define的区别，局部变量和全局变量  
this is a way of controlling  complexity  

![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP6.png)
abstraction barrier  
closure pair的元素可以是pair  

还有一点cons的底层没记笔记  

## 3A Henderson Escher Example

list pair闭包，递归scale list的例子  
map（p，l）抽象化的对list中每一个数据做操作  
你不需要去管map的实现是递归的还是循环的，you stop thinking about control structures, and you start thinking about operations on aggregates  
**for-each** does't create a copy, map 会创建一个新list  
tree recursion 不是map也不是for-each, 这里讨论的只是简单操作  

图画语言George为什么能迅速由一个primitive形成一幅画？因为闭包closure   
lisp不适合解决特定问题，但适合二级语言embed在它之上，例如George画图语言  

层级语言设计结构，操作的object在改变：点、图、对图的操作  
同等级也可以互相抽象函数，但操作的对象没有变，只是使用更高阶的procedure  
而像设计语言一样设计程序结构，可以满足高内聚低耦合。需求变更时程序员可以去决定改哪个层，怎么改，而不是像树形任务结构一样去全部剪枝重做。
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP7.png)


## Lecture 3B: Symbolic Differentiation; Quotation
符号化求道程序  
representation表示法，抽象即是separate use from representation 中间有一层抽象接口
之前课程提到的求导是单点求导，代数表达式求导公式程序的简化做引子，引用只是让+代表+而不被解释，后面可以延伸到运算符重载等。lisp自己解释自己？maybe


## Lecture 4A: Pattern Matching and Rule-based Substitution  
我们要用已构建完成的导数rules模型干什么？搞一个多用途的simplifier实践这些rules（返回一个以exp做参数的procedure例如按求导公式求导），当然也可以有其他rules集合例如乘法分配律等。  
dsimp '(dd (+x y) x)表达式前面加引用  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP8.png)
matcher和instantiater是两个不同part  

match（pat exp dic）
match过程就同时遍历两个树，判断are they the same?  
the dictionary is the answer to the match, 是一个抽象关系，我们这里没有讨论它的实现    
match是为了产出dictionary给之后的substitution（instantiation）  

instantiation实际上就是把rule左边换右边，换成dic里的真正表达式  

## Lecture 4B: Generic Operators
整节课以虚数加减乘除开始，到多项式加减乘除为止，形成一个完备的decentralized control系统通过递归可以各种套娃。  
柜子里存放的是过程，可以和别人重名甚至匿名过程，key1 polar or rect key2 real/img/mag/ang  
lets embed this in some complicated system  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP9.png)  

缺陷没有定制各个类相互之间的转换 例如3+3/7会报错，涉及coercion。但引入多项式之后各个部分又可以任意结合起来并递归地产生套娃多项式。  
如果要在顶层加新的generic operator, 需要和下面的类沟通看是否添加相关方法或部分添加相关方法。
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP10.png)  

## Lecture 5A: Assignment, State, and Side-effects（面向对象初探）
在之前交换模型概念上加入time、object的意味
assignment statement basic、pascal有但之前没讲到  
substitution model is dead，运行两次demo 3获得不同结果因为set了count的值。有点类似全局变量的结果。  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP11.png)  

新的编程模型：environment model comparing to substitution model  
所谓assignment就是set，state指procedure的状态，每个procedure对象包括代码、所属环境，当一个新的procedure object被创建时将会建立一个新环境并指向所属环境。


This has refreshed my comprehension on object-oriented programming model from once thought the pure implementation of systematic methods, which is once misled by the majority market-specific developing tools, into the representation of identities and coupling between internal states. The concept of binding variables also units with the concept of bottom-up building techniques (metaliguistic architecture) mentioned earlier in this course. Thanks for the knowledge this course has delivered.  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP12.png)
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP13.png)

## Lecture 5B: Computational Objects
modularity in programming 这是OOP的目的  
本节课我们构建和模式匹配（由lisp interpret）不一样的语言  
setxxx！（assignment）其实就是改变了对象的状态（state）
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP14.png)

agenda 优先队列  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP15.png)
由cons类似链表结构的想法，高级语言指针作为cdr取代cons作为cdr的原因，数据公用减少空间消耗，数据结构轻便，可以回指  
编程就是处理数据  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP16.png)
又一次cons、car、cdr的底层定义，运用代换试一下car（（cons x，y））  
alonzo church  
作为programmer修改一下这个定义让它也适用于更广义的environment model而不只是substitution model  

```lisp
(define (cons x y)//过程m作用在四个参数上
	(lambda (m)
		(m x
		   y
		   (lambda (n) (set! x n))
		   (lambda (n) (set! y n)))))

(define (car x)
	(x (lambda(a d sa sd) a)))

(define (set-car! x y) //x是cons，y是值
	(x (lambda(a d sa sd) (sa y))))
```

## Lecture 6A: Streams, Part 1  
环境模型更有利于分解并模拟现实中的应用场景。我们很多的课程都在介绍decompose systems。  
stream processing  

stream也是一种data structure，存在自己的selector and constructor（以及复杂数据结构的相关操作等，同时区别于抽象数据结构只有从操作和逻辑关系/selector）  
以数组整数的奇数和和树的叶子的奇数和引入，原本写程序用递归或循环写会导致enumerator、filter、map、acc的高度耦合。流做到了程序的解耦，提供conventional interfaces。  
i、j、i+j的例子和八皇后例子  
抽象问题时注意确定单个数据元素，然后决定数据元素的状态和属性（例如八皇后是格子）。  
stream和list的区别在于：若是分层enumerator、filter、acc的话，数据结构是list则每次都完全处理好再扔进下一个层。数据结构是stream则整个数据链都一直存在在各个层之间，每次需要获取下一步数据就tug on一下下层来获取（head （tail s）） **"we only compute what we need"**
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP17.png)
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP18.png)
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP19.png)

## Lecture 6B: Streams, Part 2  
long stream  
为什么可以递归定义？因为cons-stream的延时性，仅在使用的时候内部才定义
```lisp
(define ones (cons 1 (delay ones)))
```
注意区分递归程序和递归定义，但在lisp等procedure first class的语言中这二者没有区别，因为data就是procedure  
我们可能不知道什么时候应该显式地使用delay，会是一个mess，那么如何解决？  
normal-order language  
迭代state不会变，递归会变  

functional programming 放弃了assignment，解耦了time，不存在状态变化产生的冲突（局限性在银行账户服务两个人的例子）  
计算机主要的操作是computation和control，把他们严格糅合在一起不一定真的明智。

## Lecture 7A: Metacircular Evaluator, Part 1  
本节very deep三十分钟之前值得反复看
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP20.png)  
evaluateor和interpreter似乎是一个意思  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP21.png)  


理解eval和apply的相互耦合，元循环解释器/自循环解释器
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP22.png)
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP23.png)
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP24.png)
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP25.png)   


此书中的解释器包含两个主要元素：

*   Eval，即将一个[表示式](https://zh.wikipedia.org/wiki/%E8%A1%A8%E7%A4%BA%E5%BC%8F "表示式")转换为其取值的函数。
*   Apply，即将一个函数调用转换为其返回值的函数。

这两个元素互相调用，并最终将整个程序转换为其取值。
### 实例

例如以下简单的 [Scheme](https://zh.wikipedia.org/wiki/Scheme "Scheme") 表示式（假设`square`是一个内置函数，其返回值为参数的平方）：

```
(+ (square 2) (square 3))


```

其解释过程如下：

*   解释器首先将整个程序传入 Eval，而 Eval 辨认出整个程序是一个函数调用（被调用的函数是 “+”，或加法函数）。因此，Eval 会调用 Apply 来处理这一调用。
*   Apply 收到被调用函数为 “+”，参数分别为`(square 2)`与`(square 3)`。因此，Apply 分别调用 Eval 来处理这两个参数。
*   Eval 收到表示式为`(square 2)`，这是一个函数调用。因此 Eval 调用 Apply。
*   Apply 收到被调用函数为 “square”，参数为 2（注意：这里的“2” 仅仅是一个符号，而不是数字）。现在 Apply 会调用 Eval，将符号 “2” 转换为数值 2。Apply 随后调用 square 函数，并返回 4。同样地，Eval 处理`(square 3)`并返回 9。
*   现在解释器回到了 Apply 函数，处理 “+” 的调用。现在 Apply 函数有了参数的具体取值（分别为 4 和 9），并调用 “+” 函数而返回 13。
*   Eval 函数收到返回值 13，这个值是整个表示式的取值。注意以上过程中自循环解释器并没有关心具体如何实现 “+” 函数与 “square” 函数，这些细节都由底层的 Scheme 来处理。  


用一种语言实现自身的语义，这种思路叫做“元循环”（metacircular）。例如说用Java实现的JVM就是元循环JVM。  
我觉得自举挺没意义的。编译器本身就是个翻译软件，把程序翻译成机器码，仅此而已。算法相同，什么语言写的编译器结果都一样，你甚至可以人肉编译源代码，只要你使用的规则和gcc一致，那你人肉编译出来的字节码应该和原本gcc一模一样。反正机器只认字节码，至于这个字节码是不是用这门语言自举编译成的，机器并不关心。
有一件事情是肯定的，那就是自举的第一步。任何一种语言（Machine code除外）的首个compiler肯定是用其他语言写出来的，然后在其基础上再演进才能开始自举。  
Java既是编译的又是解释的，Java代码本身被编译成目标代码。在运行时，JVM将目标代码解释为目标计算机的机器代码。  
编译器只编译一次生成可执行文件，解释器每行代码编译一次不生成代码。自己都是程序。 

## Lecture 7B: Metacircular Evaluator, Part 2  
simple lexical evaluator  
dynamic evaluator  
procedure定义中的自由变量和该procedure的caller对应的变量绑定
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP26.png)

delay an argument to a procedure  

## Lecture 8A: Logic Programming, Part 1  
build a diff language  类似PROLOG逻辑编程语言的query语言（作用有点像查询？） 
* diff
* circle eval and apply
* nice use of streams to avoid backtracking  

特点：可以输入后按任意顺序输出所有可能  
primitive：query  
means of combination：and or not lispValue  
means of abstraction：rule merge-to-form  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/SICP27.png)
