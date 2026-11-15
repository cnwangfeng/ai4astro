"""Numerically verify the explicitly worked examples in revision 2.
These are deterministic teaching checks, NOT independent scientific peer review.
"""
from pathlib import Path
import json
import numpy as np
from numpy.testing import assert_allclose as eq
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from scipy.special import expit, softmax
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def record(name, result):
    checks.append({'name':name,'status':'PASS','result':result})

iris=load_iris();wine=load_wine()
eq(iris.data[[0,50,100]],[[5.1,3.5,1.4,.2],[7,3.2,4.7,1.4],[6.3,3.3,6,2.5]])
eq(wine.data[[0,59,130]][:,[0,4,9]],[[14.23,127,5.64],[12.37,88,1.95],[12.86,122,4.10]]);eq(np.bincount(wine.target),[59,71,48]);record('数据记录与类别计数','Iris三个示例记录；Wine 59/71/48')
sc=StandardScaler().fit(np.array([[1.],[2.],[3.]]));eq(sc.transform([[4.]])[0,0],np.sqrt(6))
record('标准化训练参数与新样本',float(sc.transform([[4.]])[0,0]))
x=np.array([-1.,0.,1.]);y=2*x+1; e=x+1-y
eq(np.mean(e**2),2/3);eq([np.mean(e*x),np.mean(e)],[-2/3,0])
eq(.5*np.mean((1.2*x+1-y)**2),.64/3);record('第3章一次梯度更新','a=1.2,b=1,半MSE=0.2133333333')
truth=[1,1,1,1,0,0,0,0,0,0];pred=[1,1,1,0,1,1,0,0,0,0]
eq(confusion_matrix(truth,pred),[[4,2],[1,3]])
eq([np.mean(np.equal(truth,pred)),3/5,3/4,6/9],[.7,.6,.75,2/3])
record('混淆矩阵手算','TN=4,FP=2,FN=1,TP=3')
y=np.array([2,4,6]);p=np.array([3,4,5]);eq([mean_absolute_error(y,p),np.sqrt(mean_squared_error(y,p)),r2_score(y,p)],[2/3,np.sqrt(2/3),.75]);record('回归评价指标','MAE=2/3,RMSE=sqrt(2/3),R2=.75')
x=np.array([0,1,2]).reshape(-1,1);y=np.array([1,2,2]);m=LinearRegression().fit(x,y)
eq([m.coef_[0],m.intercept_],[.5,7/6]);eq(np.sum((m.predict(x)-y)**2),1/6);record('三点最小二乘拟合','斜率.5,截距7/6,SSE1/6')
for lam in [0,1,3]:
    w=2/(1+lam);eq((w-2)+lam*w,0,atol=1e-14)
record('Ridge标量解','2/(1+lambda)；仅对应正文明确的目标归一化')
eq(expit(1),.7310585786300049);eq(-np.log([.8,.2]),[.2231435513142097,1.6094379124341003]);eq(softmax([np.log(2),np.log(3),0]),[1/3,1/2,1/6]);eq(.6*.8*.5/(.6*.8*.5+.4*.1*.5),12/13)
record('概率分类算例','Sigmoid、交叉熵、Softmax和朴素贝叶斯')
X=np.array([[1,2],[2,3],[3,3],[5,2],[5,5]]);y=np.array([0,0,1,1,1]);q=[[2,2]]
eq(np.linalg.norm(X-q,axis=1),[1,1,np.sqrt(2),3,np.sqrt(18)]);eq(KNeighborsClassifier(n_neighbors=3).fit(X,y).predict(q),[0]);record('近邻手算','前三邻居类别0/0/1，预测0')
y=np.array([0,0,0,0,1,1,0,0,1,1]);x=np.arange(1,11)
def gini(v):
    p=np.bincount(v,minlength=2)/len(v);return 1-np.sum(p*p)
def split_gini(t):
    a=y[x<=t];b=y[x>t];return (len(a)*gini(a)+len(b)*gini(b))/len(y)
eq([gini(y),split_gini(4.5),split_gini(6.5)],[.48,4/15,7/15]);record('决策树两个划分','母节点.48；4.5划分4/15；6.5划分7/15')
eq(np.mean([.51,.51,.01]),1.03/3);record('森林概率平均与硬投票','三个P(A) .51,.51,.01；平均概率预测B')
x=np.arange(1,5).reshape(-1,1);y=np.array([1.,1.,3.,3.]);f=np.full(4,2.);loss=[.5*np.mean((y-f)**2)]
for k in range(2):
    tree=DecisionTreeRegressor(max_depth=1,random_state=42).fit(x,y-f);f+=.5*tree.predict(x);loss.append(.5*np.mean((y-f)**2))
eq(loss,[.5,.125,.03125]);eq(f,[1.25,1.25,2.75,2.75]);record('两轮梯度提升',loss)
x=np.array([1.,2.]);W=np.array([[1.,0.],[-1.,1.]]);b=np.array([0.,-2.]);h=np.maximum(W@x+b,0);s=np.array([2.,-1.])@h-.5
eq(h,[1,0]);eq(s,1.5);eq(expit(s),.8175744761936437);record('MLP双隐藏单元前向计算',float(expit(s)))
def network_loss(theta):
    w1,b1,w2,b2=theta;prediction=w2*max(w1*2+b1,0)+b2;return .5*(prediction-1)**2
v=np.array([.5,0.,3.,0.]);grad=np.array([12.,6.,2.,2.]);eps=1e-6
numgrad=np.array([(network_loss(v+np.eye(4)[i]*eps)-network_loss(v-np.eye(4)[i]*eps))/(2*eps) for i in range(4)])
eq(numgrad,grad,rtol=1e-7,atol=1e-7);updated=v-.01*grad;eq(updated,[.38,-.06,2.98,-.02]);eq(network_loss(updated),.568178)
record('MLP反向梯度与数值差分',{'grad':grad.tolist(),'updated_loss':network_loss(updated)})
X=np.array([[-2,-2],[-1,-1],[1,1],[2,2]]);p=PCA(n_components=1).fit(X);eq(p.explained_variance_,[20/3]);eq(p.inverse_transform(p.transform([[1,0]])),[[.5,.5]]);record('PCA手算与sklearn',{'variance':float(p.explained_variance_[0]),'reconstruction':[.5,.5]})
X=np.array([[0.],[2.],[8.],[10.]]);km=KMeans(n_clusters=2,init=np.array([[0.],[2.]]),n_init=1,random_state=42).fit(X);eq(np.sort(km.cluster_centers_.ravel()),[1,9]);eq(km.inertia_,4);record('K-means聚类手算','中心1/9，SSE4')
db=DBSCAN(eps=.16,min_samples=3).fit(np.array([[0.],[.1],[.2],[.35],[2.]]));eq(db.core_sample_indices_,[1,2]);eq(db.labels_,[0,0,0,0,-1]);record('DBSCAN核心边界噪声','核心.1/.2，边界0/.35，噪声2')
eq((4-1)/max(4,1),.75);record('轮廓系数手算',.75)
OUT=ROOT/'review';OUT.mkdir(exist_ok=True)
(OUT/'worked_examples_checks.json').write_text(json.dumps({'scope':'deterministic numerical self-check; not peer review','total':len(checks),'checks':checks},ensure_ascii=False,indent=2))
print(f'{len(checks)} numerical check groups passed.')
