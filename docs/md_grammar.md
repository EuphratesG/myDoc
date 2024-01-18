# 这是第一级标题

## 段落

原语法是  
段落俩回车  
换行俩空格一回车

这里估计是make down enhanced升级过了，原本俩空格才换行，空两行才另起一段，升级之后自然换行就换行，空两行另起一段不变  
并且在csdn上一样可以直接导入md文件，似乎这语法升级过了，但是为了语法兼容还是用两回车换行得了

### 强调和列表

**hello a** toggle bold ctrl b  
*hello b* toggle italic ctrl i

1. 这是一级列表
2. 123123
   1. 直接按tab缩进生成二级列表  
kkk  

kkk

1. 两个列表之间要空一个段落

## 插入图片 直接ctrl v

这里装了paste image 插件

*你好吧*

## 公式

这是单独一段的公式 ctrl mm
$$
\lim_{x \to \infin}\frac{\sin(x)}{x}=1
$$

这是含在文字中的公式 $\lim_{x \to \infin}\frac{\sin(x)}{x}=1$ 嘿嘿

## 表格

小明|小红|小白
---:|:---|:---:
3岁|412313岁|11231231岁

默认是居中的

## 链接

裸链接
<https://www.bilibili.com/video/BV1si4y1472o/?spm_id_from=333.788.recommend_more_video.-1&vd_source=1e9369bdf50b21325055a7d2089c90b7>  
文字添加[链接](https://www.bilibili.com/video/BV1si4y1472o/?spm_id_from=333.788.recommend_more_video.-1&vd_source=1e9369bdf50b21325055a7d2089c90b7) 直接选中要添加的部分然后ctrl v

## 代码块

```c++
#include<bits/std++.h>

using namespace std;

int main(){

        cout<<"Hello World!"<<endl;

return 0;

}
```

## 出现的问题

1. preview enhanced 导出png无问题，导出pdf公式不显示  
可以理解成编译md文件和生成pdf是两个进程在做。之前生成pdf的速度太快了，以至于md还没编译完就生成了pdf文件。改成3秒之后就能等到编译完成了。
2. 前面说到的编辑换行formatting也换行，也是MPE插件问题，为了兼容最好还是打两空格换行


git可以用来管理纯文本文件的版本，因此markdown可以用其来管理
发博客，做成电子书