# git

commit_id代表做出这次commit的变化后的本地仓库状态  
三个区域结构一致

主分支、公有分支、个人分支  
对于公有分支来说是只能前进不能后退的，因此要撤销之前commit的影响只能使用revert生成一个新的commit再push。  
如果是个人分支使用了reset减少了commit节点数，push的时候要使用git push -f强制push才能上传到远端。公有分支不能用

## github极简工作流  

1. git clone https// 到本地
2. git checkout -b my-feature 切换至新分支my-feature
（相当于复制了remote的仓库到本地的xxx分支上，本地目前有两个分支）
3. 修改或者添加本地代码（部署在硬盘的源文件上）
4. git diff 查看自己对代码做出的改变
5. git add 上传更新后的代码至暂存区
6. git commit 可以将暂存区里更新后的代码更新到本地git
7. git push origin my-feature 将本地的my-feature分支上传至remote（在remote也会创建一个新的my-feature分支）

-----------------------------------------------------------
（如果在写自己的代码过程中发现远端GitHub上代码出现改变）

1. git checkout main 切换回本地main分支（原汁原味之前clone的）
2. git pull origin master/main 将远端别人修改过的代码再更新到本地main
3. git checkout my-feature 本地回到my-feature分支
4. git rebase main 我在my-feature分支上，先把main移过来，然后根据我的commit来修改成新的内容接在main别人的新改动后面
（中途可能会出现，rebase conflict -----》手动选择保留哪段代码）
1. git push -f origin my-feature 把rebase后并且更新过的代码再push到remote上
（-f ---》强行，做了rebase的后果）
1. 对原项目主人要求做pull（本地pull request）项目主人一般选择squash and merge 合并my-feature分支上所有不同的commit成一个commit再挂在main分支后面。

-----------------------------------------------------------
远端完成更新后
1. 项目主人直接删掉my-feature这个分支
2. 本地切换到main上
3. git branch -D my-feature 在main上操作删除本地的my-feature分支
4. git pull origin master/main 再把远端的最新代码拉至本地



##mess

git官方文档也不说人话，且--keep、--merge少用，不再深究  

完成简悦下载，打开文章后ctrl+r进入阅读模式可以将网页转markdown下载。但是部分网站好像存在一些延迟？不管了  
![](https://cdn.jsdelivr.net/gh/EuphratesG/myPic@main/git1.png)

硬盘对于源文件来自哪个branch一无所知