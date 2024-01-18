# algorithm  
1. 数组
2. 字符串
3. 链表
4. 数学
5. 哈希
6. 二分
7. 栈
8. 双指针
9. 贪心
10. 回溯
11. dp
12. dfs
13. 树
    
## PageRank google对网页重要性rank的算法  迭代算法
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@master/paper1.png)  
阻尼系数d代表用户继续浏览的统计概率，也即迭代继续进行必要的损失。  
本质是越重要的网页被从周围网页跳转而来的次数越多。无论初始rank值如何，在随时间多轮迭代或设置停止条件后各个网页的PR值会收敛。  

SSSP单源最短路径算法（Single Source Shortest Path）  
不就是迪杰斯特拉？  

## 图遍历dfs，bfs  
所谓搜索也就是记录当前状态和寻找下一状态。  
dfs通常递归实现，函数参数定义状态，递归跳转状态，找到解就over。先探到底再向上回溯，中间可能会产生大量无效探索，需要剪枝策略。  
bfs通常队列实现，队列中每个元素对应一个状态，转移状态就操作队列出队入队，搜索顺序是逐层的。  
另外这两种方法本质上其实都是穷举，算法时间复杂度和具体使用的数据结构有关。   

## 单源最短路径  
### 迪杰斯特拉
1. 设置dis数组，dis[i]表示起点start到i的距离。
2. 从点集V中**弹出一个dis值最小**且**未以该点为起点进行松弛操作的**点。
3. 从该点松弛与其领接的各个点更新dis数组，返回S2，循环进行。  

```c++
//O((V+E)logV) 若不用优先队列则是O(VE)
#include<bits/stdc++.h>
using namespace std;
const int maxn=1e5+5;//点的集合范围
const int maxm=2e5+5;//边的集合范围
​
struct edge{
    int to,cost;
};
​
struct node{ //重载<方便优先队列
    int num,dis;
    bool operator < (const node &x)const{//tm必须重载<是因为priority_queue第三个参数是less<T>，相当于重载<
        return x.dis < dis;
    }//当比较器返回true时第一个参数在前面第二个参数在后面（队列从队头pop所以这样优先级高的大）
};
​
vector<edge> e[maxm];
int dis[maxn];
bool vis[maxn]; // 布尔型vis数组
int n,m,s,cnt=0;
priority_queue <node> q;
​
void add_edge(int u,int v,int w) // 存边
{
    e[u].push_back((edge){v,w}); //匿名类省去起名
}
​
void dijkstra()
{
    dis[s]=0;
    q.push((node){s,0});
    while (!q.empty())
    {
        node tmp=q.top();
        q.pop();
        int x=tmp.num;
        if (vis[x]) continue;
        vis[x]=1;
        for (edge k:e[x]) // 遍历边信息，请注意洛谷请选用c++11，否则编译错误
        {
            if (dis[x] + k.cost < dis[k.to])
            {
                dis[k.to]=dis[x]+k.cost;
                if (!vis[k.to]) q.push((node){k.to,dis[k.to]});
            }
        }
    }
}
​
int main()
{
    cin>>n>>m>>s;
    int i;
    for (i=1;i<=n;i++) dis[i]=0x7fffffff;
    for (i=0;i<m;i++)
    {
        int u,v,w;
        cin>>u>>v>>w;
        add_edge(u,v,w);
    }
    dijkstra();
    for (i=1;i<=n;i++) cout<<dis[i]<<" ";
    return 0;
}
```
由此引出自定义比较函数（通常用于结构体）的方法：  
1. sort()
```c++
//按照指定的 comp 排序规则，对 [first, last) 区域内的元素进行排序
void sort (RandomAccessIterator first, RandomAccessIterator last, Compare comp);
//首先我们要写一个bool类型的方法，用来返回参数的比较结果
//当比较器返回true时，第一个参数放在前面，第二个参数放在后面，即位置不变
//当比较器返回false时，为两元素交换位置
//这里要注意对参数相等时的处理,因为可能会在两者相等的时候交换位置，在一些特定的环境下会导致题解出错
//比较器最好写成static函数
//比较器的值可以使用引用类型,节省空间开销
static bool cmp1(int &lhs,int &rhs)//升序
{
	return lhs<rhs;
}
static bool cmp2(int &lhs,int &rhs)//降序
{
	return lhs>rhs;
}

bool comp1(Test A,Test B){
     if(A.num==B.num) return A.str<B.str; // 字符串str字典序小的在前
     return A.num > B.num ;               // 出现次数高的元素在前
}

sort(arr,arr+10,cmp1);//升序
sort(arr,arr+10,cmp1);//降序
```
2. 优先队列
```c++
//优先队列自定义比较器处理方法1，处理方法2是戴上镣铐重载<
bool cmp(vector<int>&a,vector<int>&b){
	return a[0]>b[0];
}
priority_queue<vector<int>,vector<vector<int>>,decltype(&cmp)> pq(cmp); //注意队列pq还要传入cmp
```
## 贪心、动态规划、回溯、分治 实际上属于Algorithmic Paradigm
四种编程范式都是基于递归（或有其他实现方式）之上的，而迭代和递归是并列概念编程范式Programming Paradigm  
分治只能递归实现，分治是递归的充分条件。  
动态规划两层for或者递归实现


动态规划
自底向上
分治算法要求子问题相互独立互不影响，而动态规划的子问题往往是不独立的常常适用于有重叠子问题和最优子结构性质的问题。
求最优性质
最优子结构
重叠子问题
无后效性，“无后效性”的定义是：如果给定了某一阶段的状态，那么在这一阶段以后过程的发展不受这个阶段以前各阶段状态的影响。

贪心是自顶向下的


分治和dp的区别是子问题之间独立，且最好能两两分，不要求最优子结构

减治（也是朴素递归的一种）就是缩小问题的规模，比如说规模为 n 的缩小的 n-1 来求解，典型的如插入排序，对 n 个数的排序，就是先对 n-1 个数进行排序，然后再求解小规模到大规模的解。
分治是把问题划分成很多相似的子问题，比如 n 个数划分为 3 个 n/3 规模的数，子问题和原问题的类型相同，求解方式一样，故常递归来解。
两者并不相同。  


## 链式前向星 
前向星实际上是图以类似邻接表的形式存储的一种数据结构。  

为什么用结构体指针数组而不是用结构体数组？空间复杂度  
