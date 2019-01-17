
# coding: utf-8

# In[60]:


"""
以下为绘制一颗二叉树
"""

import matplotlib.pyplot as plt
import pickle


# In[114]:


decisionNode = dict(boxstyle="sawtooth", fc="0.8")# 绘制文本框和箭头样式
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="->")


# # plotNode
# 
# plotNode绘制带注解的箭头，由于决策树绘制时需要内嵌带箭头的划线工具，可以从父节点指向数据位置，并在数据位置添加描述信息。
# 
# plotNode需要四个参数，描述信息、数据位置、父节点位置和节点类型。
# 
# 节点类型包括决策点和结果点，不同结点类型的绘制标签不一样。
# 
# arrowstyle为划线样式，当前为$->$，可以为其他类型不过要参照文档，不能自定义。

# In[111]:


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',
                           va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)
    
def createPlot():
    fig = plt.figure(1, facecolor='white')
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False 
    fig.clf()
    createPlot.ax1  = plt.subplot(111, frameon=False)
    plotNode("Node", (0.5,0.1), (0.1,0.5), decisionNode)
    plotNode('Leaf', (0.8,0.1), (0.3,0.8), leafNode)
    plt.show()

createPlot()


# In[94]:


def getNumLeafs(myTree): # 获取叶子数
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):  # 获取决策树深度
    maxDepth = 0
    firstStr=list(myTree.keys())[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]


# # 绘制决策树图像
# 绘制原理也很简单，就是给定一个初始位置作为根节点，决策树绘制时要保证各个节点之间的动态距离所以要知道决策树深度和宽度。
# 
# 深度好理解，就是决策树的层数，而宽度没法直接获取而用叶子数间接表示。
# 
# 绘制图时首先绘制根节点，然后计算子节点的位置，计算方法如下：
# 
# plotTree.xOff用来保存当前节点的横坐标
# 
# plotTree.yOff用来保存当前节点的纵坐标
# 
# plotTree.totalW是树的宽度
# 
# plotTree.totalD是树的深度
# 
# 计算当前节点位置：
# 进入函数是决策节点，所以横坐标位置为之前横坐标的值加上当前树的宽度除以２再除以树的宽度，为什么要这么算呢?
# ![](data:image/png;base64,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)
# 
# 参考上面的图会发现子节点的横坐标是父节点横坐标加上子节点树的宽度除以２,如何算子树的宽度呢？子树的宽度是一个动态变化的值。 1+float(numLeafs)代表当前子树需要绘制的宽度(1代表根节点),为什么要除以总树的宽度呢，是因为要对子树的实际长度进行一下放缩，如果不放缩就会导致子树之间互相覆盖，这就是子树根节点横坐标的计算方式。
# 
# 细心的朋友还会发现在绘制决策点是没有改变plotTree.xOff的值，但是在叶子节点上却修改了，这是为什么呢。可以假设一下改变会发生什么，如果决策点改变的话，那么子节点的坐标都会在叶子节点右边，现在可以看出其实plotTree.xOff代表当前叶子节点横坐标最新的参考值，其他点都需要参考值得到当前点的位置。

# In[141]:


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
             xytext=centerPt, textcoords='axes fraction',
             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
def plotMidText(cntrPt, parentPt, txtString): # 计算父节点和子节点的中心位置，添加文本信息即0或１
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on  深度优先遍历
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)  #计算当前节点的位置
    plotMidText(cntrPt, parentPt, nodeTxt) 
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():  # 递归绘制图形
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


# In[142]:


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks
    #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses 
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW 
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5,1.0), '')
    plt.show()

