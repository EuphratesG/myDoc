# mess
anaconda不建议加入系统环境变量，说可能会和其他应用产生冲突，因为base环境会覆盖，所以建议使用anaconda prompt
pip是Python的默认包管理工具，可以从Python官方仓库PyPI（Python Package Index）中下载和安装Python包。而conda是Anaconda的包管理工具，可以从Anaconda仓库中下载和安装Python包，同时也可以管理非Python的包。

直接在vscode设置里加入默认虚拟环境文件夹G:\anaconda3\envs，可以找到anaconda解释器，但不是通过anaconda prompt运行的文件  

Could not fetch URL https://pypi.mirrors.ustc.edu.cn/simple/mkdocs-material/: There was a problem confirming the ssl certificate: HTTPSConnectionPool(host='pypi.mirrors.ustc.edu.cn', port=443): Max retries exceeded with url: /simple/mkdocs-material/ (Caused by SSLError(SSLZeroReturnError(6, 'TLS/SSL connection has been closed (EOF) (_ssl.c:1131)'))) - skipping

源的ssl证书有问题，直接trust即可
pip install mkdocs-material -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

pip只能管理python包，conda还能管理其他包，有论调说conda全面代替pip  
安装conda的git，安装在虚拟环境myBlog下，执行命令git clone git@github.com:EuphratesG/clearlove.git复制仓库，出现Git使用出现git@github.com: Permission denied (publickey)错误。
要给git设置和github一样的username和email（但是只是在虚拟环境下不知道会不会有啥问题）参考csdn：<https://blog.csdn.net/qq_43768946/article/details/90411154?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164837496016780274125034%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=164837496016780274125034&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2>


github就是基于git这个版本控制软件的仓库托管的网站

git三个概念 commit repository branch  



anaconda只针对python设置的环境，而且在vscode select interpreter中选中哪个为全局，下次运行py文件时就会自动切换到那个环境，包括本机自带的python环境和base环境。

很有意思，修改注册表让右键出现装在conda\pkgs的git bash here  
https://blog.csdn.net/qq_41559271/article/details/115800774  
git clone "github上的.git链接" 即可将github上的仓库拉至本地

项目的license指的是将原项目商用需要遵守的，mit比较轻只要保留原作者版权信息就可以无限使用


master和main分支 master涉及种族歧视之前用的

建立了myBlog仓库之后还是报这个错
this directory does not appear to be a git repository

现在全栈开发少了，都是前后端分离，分工明确，一个项目流程基本分为：产品经理对接需求，做需求分析，画原型，美工根据原型做ui切图，前端程序员写前端页面做数据交互，后端程序员写服务端业务逻辑提供接口，测试人员做代码功能安全测试，实施人员搭配生产环境做项目部署，运维人员做日常维护工作。


我靠这个老早的帖子还可以推送给我，我真的有这么菜？前后端分离否，看看controler 是return json 等data ，还是forward/redirect some. jsp 应该就可以了吧。前段服务器中存储了大量的html. 基本不用jsp的啦
说下我的定义， 
1.假的前后端分离，生成html的服务器与数据库有链接。
2.真正的前后端分离，生成html的服务器与数据库没有链接！
网页一个服务器，后端一个服务器连数据库，判断前后端是否分离的关键就是传给浏览器的是网页还是数据。

github pages和自己建站  
博客平台和独立博客分别有啥区别？


mkdocs作用
在git平台创建mkdocs主题仓库，自动将markdown文件生成静态网页。 


也就是说，可以直接本地一条命令更新网页部署，只不过需要清楚浏览器缓存才能显示或者正常中文检索。  
可以写个脚本定期将项目源docs推到github上保证云版本

这样我们每次只需要在本地更新内容之后，通过该命令进行提交、部署，就可以实现文档内容的更新。

注：由于 Github Actions 运作需要一定时间（大约在几分钟到十分钟不等），因此部署完成后你需要等待一会儿才能看到 Github Pages 生效，否则会出现 404 错误。

这个分支里面是根据docs目录与mkdocs.yml这两个生成的网站的静态资源！


环境变量其实就是让任何命令行都能直接找可执行程序的一个列表，不用cd到特定bin目录之下就可以执行内部或外部命令、可运行的程序或批处理文件.系统变量就是都可以用的，用户变量就是那个用户可以用的。

博客整体部署流程：
1. 先本地文件夹部署再和空仓库建立连接/或者直接空仓库克隆下来估计更好。强行合并会导致一些冲突
2. 如下
   
## GitHub Pages
----

If you're already hosting your code on GitHub, [GitHub Pages](https://pages.github.com/) is certainly the most convenient way to publish your project documentation. It's free of charge and pretty easy to set up.

### with GitHub Actions[¶](#with-github-actions "Permanent link")

Using [GitHub Actions](https://github.com/features/actions) you can automate the deployment of your project documentation. At the root of your repository, create a new GitHub Actions workflow, e.g. `.github/workflows/ci.yml`, and copy and paste the following contents:

```
name: ci 
on:
  push:
    branches:
      - master 
      - main
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: 3.x
      - run: echo "cache_id=$(date --utc '+%V')" >> $GITHUB_ENV 
      - uses: actions/cache@v3
        with:
          key: mkdocs-material-${{ env.cache_id }}
          path: .cache
          restore-keys: |
            mkdocs-material-
      - run: pip install mkdocs-material 
      - run: mkdocs gh-deploy --force


```



1.  This step is only necessary if you want to use the [built-in optimize plugin](https://squidfunk.github.io/mkdocs-material/setup/building-an-optimized-site/#built-in-optimize-plugin) to automatically compress images.
    
2.  Remember to set the `GH_TOKEN` environment variable to the value of your [personal access token](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token) when deploying [Insiders](https://squidfunk.github.io/mkdocs-material/insiders/), which can be done using [GitHub secrets](https://docs.github.com/en/actions/configuring-and-managing-workflows/creating-and-storing-encrypted-secrets).
    

Now, when a new commit is pushed to either the `master` or `main` branches, the static site is automatically built and deployed. Push your changes to see the workflow in action.

If the GitHub Page doesn't show up after a few minutes, go to the settings of your repository and ensure that the [publishing source branch](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) for your GitHub Page is set to `gh-pages`.

Your documentation should shortly appear at `<username>.github.io/<repository>`.

另外本地可以直接 mkdocs serve在本地看网页效果。有时间可以写个定时推送脚本？maybe。想要上传了直接push到github就可以自动更新网页

material是一个mkdocs主题

现存问题：

1. 各种终端的区别 anaconda默认是cmd command prompt
2. base环境对非python有没有影响 √
3. 部署网站到github并实现commit等 √
4. 翻译软件
5. 获取网页markdown的插件简悦 √
6. pdf处理  



但归根到底几个充要条件其实都表达了一个意思，就是空间V可以表示为矩阵所有特征子空间的直和时，必可以对角化。  
T关于所有λ的特征子空间都是T的不变子空间  
线性变换的特征值和对应矩阵的特征值相同，**由于定义上线性变换的特征值是由其矩阵的特征值和特征向量计算出来的**，因此哲学上线性变换的特征向量不变，但其表示是跟随矩阵特征向量而变的（因为基不同）。  
同一个特征值λ对应的特征子空间可能是多维的，这一点还没法直观解释，知道就好。  
怎么求一个线性变换在某组基下的矩阵？记住矩阵每列都是对对应基向量操作后的结果在这组基下的坐标，也就是说矩阵里填的值就是对基做线性变换的结果向量坐标。  




极大似然估计（MLE）是一种统计方法，用于从数据（样本）中估计模型(分布)的参数   
本质就是function中的每个f分别对变元中的每个元素逐个求偏导，求导的布局是另外考虑的事情。**先算出结果再去考虑布局。** 分子布局就是分子是没转置的列向量，分母是转置之后的行向量或者矩阵。矩阵变元和矩阵函数都用vec拉成向量形式去解决。
摆放：列优先去求导，行优先去放在结果矩阵里面。即所谓先计算再摆放。       
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@main/202311101033658.png)  
这里如果按先计算的原则的话xtA和Atx是等价的。
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@main/202311101050211.png)
如果这个了然了，矩阵求导也就了然。  
but！神经网络动不动就几百层，你去手写导数是一件很难的事情。  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@main/202311101057746.png)
这里正向先给出各个中间变量的表示，反向再去用这些表示去求梯度。  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@main/202311101102873.png)

**深度学习中我们很少用向量函数求导，绝大多数对标量求导**。即便y有些时候是向量通常也会把它转化成标量（例如sum()一下,是向量每个分量的函数，但是求偏导全没了）再去对向量求导。    
具体来说，.detach() 的作用是返回一个新的张量，该张量与原始张量共享相同的底层数据存储，但不再与计算图相连接。（也就是说不再参与链式法则）例子：
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@main/202311101114988.png)
深度学习框架可以自动隐式地构建计算图，因此如果x定义时确定了要算梯度，那么得到y之后调用一个backward函数就可以得到y关于x的梯度。这种自动求导还有一个好处就是可以跨越python控制流得到梯度。一个python函数f(a)通过反向传播也可以获得a的梯度。  

**上午断章在线性回归视频，大概已经完全参透线性回归了，留个尾巴给简要表示**

线性回归一般是要用均方损失和sgd（随机梯度下降），softmax（逻辑回归）则用log似然？看看  
注意最后sgd是对[w,b]进行的更新，也就是说b偏置也要更新。  
注意课程里给出的构建人造数据集的过程，是已知了真实的w和b再通过这个去构建样本特征和标签。而后面还会通过模型将另一组随机初始化的w和b优化成近似真实值的w和b。所谓true值只是为了数据集的构建方便，和最后能对模型训练出的w和b与真实值做一个评估。好像qinbin也让我找看看有没有真实数据集，可以看看。  

公私钥属于非对称加密，对称加密指加密解密都用一个钥匙，但是这样钥匙容易被截取，因此通常用非对称加密去加密钥匙。  
每个人都有自己的公钥和私钥，但私钥只有自己知道内容。  
公钥加密私钥解密别人只知道锁没有钥匙，对我自己的隐私数据加密。    
私钥加密公钥解密，把公钥公开给服务器，这样要远程连接的时候服务器给客户端发一个字符串，客户端用私钥加密后只有用之前保存的公钥解密才能还原出原来的内容。因此私钥加密相当于身份验证代替用户名密码。  


负采样的思想是通过仅考虑少量的负样本来近似全局损失，从而提高了训练效率，尤其是在大规模词汇表的情况下。  
可以不用sudo就不用sudo的，就用chmod加权限就行了  
fl场景下的KGembedding并且用于KGcompletion  
由所谓data isolation引入，传统FedAvg里面各个用户的训练数据不会传来传去   
这篇文章搞的是embedding而已，保留语义信息  
通过实体对齐share实体的embedding  
FEDE每个用户维护一个KG，实体有重叠但是不知道其他用户KG的关系和三元组（也就是说实体也不知道）。和fedavg一样本地训练KGE之后由server去聚合本地训练的结果。本文还增加了本地用户获得聚合的结果之后的一个模型聚合过程（决定使用自己训练的结果和聚合结果的比例）  
本文主要干了：①提出联邦setting下的KGcompletion作为task②提出框架FEDE联邦对实体进行embedding，并用于前述KGC任务是。③实验证明和参数分析  


fede主要是搞出来联邦场景下的embedding，至于说用于补全则是老生常谈三种预测，给定头尾然后关系一个一个试。譬如对于关系预测问题，**直观来说大家或许会认为应该是输入头节点和尾节点**，对关系进行分类。但是知识图谱中的关系往往很多，很多是几千，这样大数量级的多分类问题又会带来类别不平衡等问题，并且训练复杂。因此，**目前的方法多是采用打分的机制**，（所谓打分函数）对于一个三元组，我们给出这个三元组可信的评分。在关系预测问题中，就是给定头节点和尾节点，在所有待选的关系集合中，选出评分最高的作为关系预测的结果。  



把注意力集中到数据的理解、清洗、预处理、人肉特征、业务应用（而这些往往和屌丝、苦逼等形容词联系在一起）上来  

背景、目标、拟研究问题、假设、研究方法、预期结果、时间安排、预期调整、参考文献等部分的言之有物、逻辑清晰的 proposal，那么恭喜你，考虑继续读个博吧  
软件不是目的，而是手段。人的需求才是目的。  


第三方平台审核较为严格，可以自己搭建自己的博客去分享  
nginx其实就是类似cache的在客户端和服务器之间的一个高速web服务器。  

路由器还能ssh登录？可以看一看  


注意softmax回归的交叉熵损失函数的标签可以不是oh编码而是一个概率分布，但这样不影响交叉熵损失的计算  



主要针对的都是不同用户之间的重叠实体，才有利用的价值，fede figure1的例子  
KGE methods [3, 23, 25, 31]  
fede client本地怎么更新？用已有的KGE方法，打分函数（需要用到embedding）结合负采样构成损失函数训练    
负采样损失函数  
明天看fede代码  
事实上，交叉熵loss原本就是取ground truth类别的预测概率的负对数作为loss，因为真实分布中只有真实类别（ground truth类别，或者就叫正例）的标签为1，其他（负例）全为0。 

负样本打分越高的负样本权重越高，理解成要去着重优化降低其打分。整个损失函数理解成提高正样本打分，降负样本打分


1226  
这段代码是在一个深度学习的环境中设置随机种子，这样做的目的是为了确保在每次运行模型时都能得到相同的随机数序列，以便于结果的可复现性。

- `np.random.seed(args.seed)`: 这行代码使用NumPy库来设置随机数生成器的种子为`args.seed`，这样在使用NumPy生成随机数时，每次运行时都会得到相同的随机数序列。

- `torch.manual_seed(args.seed)`: 这行代码是使用PyTorch库来设置随机数生成器的种子为`args.seed`，确保在使用PyTorch进行训练和操作时，得到的随机结果也是可重复的。

- `torch.cuda.manual_seed(args.seed)`: 这行代码是设置PyTorch在GPU上生成随机数的种子为`args.seed`，这样在使用GPU加速训练时也可以保证结果的可重复性。

通过这些操作，你可以在每次运行深度学习模型时获得相同的随机数序列，这对于调试和验证模型的性能非常有用，因为可以确保不同运行下的结果是一致的。

main函数训练的时候构建的是trainer对象，trainer内部会构建KGEModel对象。 
打分函数就在KGEModel那几个函数里面啊，我直接换成LLM不就行了。  
找到了Llama2的论文先看看，说是能达到gpt3的水平  
a family of pretrained and fine-tuned LLMs, Llama 2 and Llama 2-Chat    
我在Meat的官网上看到 llama2 是构建在PyTorch之上的，而ChatGPT是基于TensorFlow Probability框架的，本文里面就简称TFP。  
LLaMa 的网络基于 Transformer 架构，在此基础上做出一些改进。  

目前FedAvg这个环境应该是fede、llama都能跑  
环境换成了cuda11.8，torch2.1.2，torchvision0.16.2，报英伟达驱动版本太低  
什么叫监听端口，服务器才会监听端口等待数据的传输  
AttributeError: module 'torch' has no attribute 'inference_mode' 需要torch1.9以上  
换！cuda11.8，torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1

1/8  
预训练模型的最终目标和输出取决于在微调阶段应用它的具体任务。预训练的语言模型，比如BERT、GPT等，在预训练阶段通常执行自监督任务，比如掩码语言建模（masked language modeling）。在微调阶段，这个预训练模型被用于特定的下游任务，比如文本分类、命名实体识别、机器翻译等。
文本分类：模型的输出可能是一组类别标签的概率分布，用于对文本进行分类。  
命名实体识别：输出可能是对文本中实体的边界和类型的预测。  
机器翻译：输出可能是对源语言句子的翻译或目标语言句子的生成。  
掩码语言建模输入带掩码的句子（词标号序列）和被masked的词作为自监督的标签，训练模型去预测这个词，模型参数就是各个词的embedding。  
自回归模型其实类似于n-gram例子，就是预测下一个出现词的概率，且下一个出现词的概率只和前面若干个词有关。  

TransE的损失函数理解损失函数是使用了负抽样的max-margin函数。L(y, y’) = max(0, margin - y + y’)y是正样本的得分，y'是负样本的得分。然后使损失函数值最小化，当这两个分数之间的差距大于margin的时候就可以了(我们会设置这个值，通常是1)。优化目标是往0去优化。这里的d是特征向量空间中h+l和t要很接近，负样本不接近，那么就会产生一个负值和γ相加，**属于自监督学习。**    
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@main/202401081126322.png)

服务器没法翻墙怎么办，直接搞了个镜像网站export HF_ENDPOINT=https://hf-mirror.com  


目前找到了用NN的打分函数，torch环境没配好，明天做做ppt，看看llama，看看树的加速  

树的加速两篇论文，解释在txt里，llama似乎输入和输出固定是文本的，网上只找到了获得输入文本的embedding的方法（输入text输出embedding），那么没找到怎么把输入换成embedding？  

Specifically, the neighbor features of an entity are first aggregated under each relation-path. Then the importance of different relation-paths is learned through the relation features. Finally, each relation-path-based features with the learned weight values are aggregated to generate the embedding representation.

huggingface的pipeline可以选择各种任务  
llama是一个decoder-only的结构  
tokenizer方法是可以把tokenID转化成pytorch张量的，也就是embedding呗  


可以把多层感知机能够拟合异或的原因理解成低维空间线性不可分那么高维空间线性就可分了，是支持向量机核函数的思想。  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@main/202401181602663.png)


1/11 发现llama把embedding的接口隐藏了啊，暴露出来的只有token啊，看看fede的代码  
自循环的模型似乎只能接受token作为输入（generate的输入）而不能接受embedding，但似乎可以把自己的embedding转化成token  
另外这样想来既然可以截胡LLM的embedding的话那么在生成embedding的时候就可以加入文字描述作为权重了，打分函数那个应该只能把embedding变成token，但这样又很怪    
现在还需要把输出变成概率的形式  
 Logits are often used for calculating probabilities through softmax activation, while embeddings serve as input representations for the model.  
 randn是你给格式生成高斯随机数，tensor是你给数据他转成tensor  
模型最后生成的token提供了获取logits的接口，那这样直接把logits扔到softmax里面应该可以？看看fede的论文和代码结构怎么说。  


fede实际上就是先进行t个round，每个round随机在所有client里面选一部分做联邦。全部收敛之后再对每个用户model-fusion权衡本地embedding（isolation）和联邦得来的embedding的打分。和collection的对比中FEDE只是部分占优势，但是相比其优点在于保证了信息的隐私（三元组不被传播至传输embedding）    
In a typical Large Language Model (LLM) scenario, the model expects tokenized input, not raw embeddings.  
So, in essence, the logits produced by the LLM can be interpreted as a form of scores, indicating the model's confidence or belief about the likelihood of different tokens in a given context.  

pylance速度很慢的原因原来是workspace文件太多卡住了，重新按小文件夹打开就好了正常显示补全了。  

断章在llamaForCasualLM和automodelForCasualLM有啥区别  
<class 'transformers.models.llama.modeling_llama.LlamaForCausalLM'>这是GYHtest里面model的输出，换成autoModel发现还是这个输出，因此可能有可能从这个类里面找到输入embedding的可能性。  

今天发现输出改成概率是好弄的，明天看看上面说的那个类，还有fede里面怎么把LLM在训练embedding里面也用上。  

看了llamamodel的pytorch结构发现原来是可以直接输入embedding的，我真佛了，还是要看模型结构啊  

最新成果，首先可以把embedding当输入，然后输出的话可以看看llamaModel的输出层结构考虑怎么由logits后加一个softmax转换成概率/打分  
贝叶斯决策就是在不完全情报下，对部分未知的状态用主观概率估计，然后用贝叶斯公式对发生概率进行修正，最后再利用期望值和修正概率做出最优决策。**发生证据的条件下对某个全概率事件的可信度。**  
至于条件概率是很自然的要乘条件时间的概率再乘条件下的概率的  


1/15最新成果，发现了输出的logits是torch.Size([1, 13, 32000])，第一个是批batch_size因为输入就一句话，第二个是根据句子截出来的时间步，第三个是线性层输出。得到了一个打分函数的值，下一步就是把输出改成embedding跑fede，这个要琢磨琢磨。llama原本embedding层是32000、4096，所以估计用自己的embedding输入的时候就用4096维的向量就好。（但是fede里面是256维的，这个思考一下）  
fede先是用了两个评估知识图谱的数据集，又针对联邦的evaluate搞了新的数据集。  
联邦的过程是先随机分给用户relation，再去匹配对应的实体。把前述两个数据集分别分给3510个client和3个client，训练集、验证集、测试集8：1:1.  

model_fusion是和fede单搞出来的，我们拿来做改进的话可以不用改。  

把dataloader直接赋值给三个变量的操作是因为这样做调用了dataloader所对应数据集的getitem方法的返回值  

nem_neg负采样样本的数量可以自己制定？cool  


B站学尚硅谷所有后端课程 一直到谷粒商城

然后包装简历 别写商城和外卖

搞点github上的开源项目到简历上

然后背八股 可以买一份八股文档

如果学有余力可以接着学前端的js之类的 是个加分项

八股要熟

剩下的看天命

预处理（include换成真文本，仍然是文本） 编译（狭义编译成汇编） 汇编（assemble，.h文件也会产生.o目标文件） 链接（把预处理的那些链接起来，再加上库函数一起链接得到可执行文件linux和windows不一样）   
而java则先编译成.class文件   

123123123123123123123