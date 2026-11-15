"""Regenerate all analytical PDF and PNG figures for the textbook.
All figures are based on sklearn built-in datasets or explicitly synthetic examples.
No unverified observational/experimental results are drawn.
Run: python scripts/generate_figures.py
"""
from pathlib import Path
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.datasets import (load_iris, load_wine, load_digits, load_diabetes,
                              make_moons, make_blobs)
from sklearn.model_selection import (StratifiedKFold, KFold, cross_validate,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingRegressor, IsolationForest)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning
# ConvergenceWarning is intentionally visible: an iteration limit is not convergence.
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'figures'; DEST.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'axes.titlesize':11,'axes.labelsize':10,
                     'figure.dpi':110, 'savefig.dpi':170})
GENERATED=[]
def save(name, fig, dpi=160):
    fig.savefig(DEST/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(DEST/f'{name}.png',bbox_inches='tight',dpi=dpi)
    plt.close(fig)
    GENERATED.append(name)
def box(ax,x,y,w,h,text,fc='#F1F5F9'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035',linewidth=1.1,
                                edgecolor='#607b8a',facecolor=fc))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(ax,x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',
                                mutation_scale=13,linewidth=1.2,color='#506475'))
def workflow(name,titles):
    fig,ax=plt.subplots(figsize=(10.2,2.5)); ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis('off')
    n=len(titles);w=(9.1-(n-1)*0.18)/n; start=0.45
    for i,t in enumerate(titles):
        x=start+i*(w+0.18); box(ax,x,0.73,w,0.65,t)
        if i<n-1: arrow(ax,x+w+0.02,1.055,x+w+0.16,1.055)
    save(name,fig)

iris=load_iris(); wine=load_wine(); digits=load_digits(); diabetes=load_diabetes()
Xi,yi=iris.data,iris.target
Xw,yw=wine.data,wine.target
Xd,yd=diabetes.data,diabetes.target
# Chapter 1
workflow('ch01_workflow',['Question','Data','Split','Train + CV','Freeze model','Final test','Interpret'])
# Chapter 2
fig,ax=plt.subplots(figsize=(7.5,4.1))
for k,t in enumerate(iris.target_names):
    sel=yi==k;ax.scatter(Xi[sel,2],Xi[sel,3],s=17,alpha=.8,label=t)
ax.set(xlabel='Petal length (cm)',ylabel='Petal width (cm)',title='Iris: two of four measured features');ax.legend();fig.tight_layout()
save('ch02_iris_scatter',fig)
fig,ax=plt.subplots(figsize=(9,4.0));
std_before=np.std(Xw,axis=0);std_after=np.std(StandardScaler().fit_transform(Xw),axis=0)
x=np.arange(len(std_before));ax.plot(x,np.log10(std_before),marker='o',label='Raw log10(std)')
ax.plot(x,np.log10(std_after),marker='s',label='Scaled log10(std)')
ax.axhline(0,color='0.6',lw=.6);ax.set(xlabel='Wine feature index (0-12)',ylabel='log10 feature standard deviation',title='Feature scales before and after standardization')
ax.set_xticks(x);ax.legend();fig.tight_layout();save('ch02_wine_scales',fig)
# Formal predictive figure comparisons now share the 80% development split.
# Original table exploration above uses all records; no labels enter its transformations.
Xi, Xi_hold, yi, yi_hold = train_test_split(Xi, yi, test_size=.2, stratify=yi, random_state=42)
Xw, Xw_hold, yw, yw_hold = train_test_split(Xw, yw, test_size=.2, stratify=yw, random_state=42)
# Chapter 4
fig,ax=plt.subplots(figsize=(10,2.6));ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis('off')
box(ax,.5,.75,2.5,.6,'Whole dataset')
arrow(ax,3.05,1.05,4.05,1.5);arrow(ax,3.05,1.05,4.05,.6)
box(ax,4.1,1.25,2.3,.55,'Development 80%')
box(ax,4.1,.3,2.3,.55,'Final holdout 20%')
arrow(ax,6.5,1.5,7.25,1.5)
box(ax,7.32,1.25,2.1,.55,'Cross-validation')
ax.text(7.3,.5,'Never tune on holdout',fontsize=10)
save('ch04_split',fig)
Xtr,Xte,ytr,yte=train_test_split(Xi,yi,stratify=yi,test_size=.25,random_state=42)
clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(Xtr,ytr)
cm=confusion_matrix(yte,clf.predict(Xte))
fig,ax=plt.subplots(figsize=(5.4,4.5));ConfusionMatrixDisplay(cm,display_labels=iris.target_names).plot(ax=ax,cmap='Blues',colorbar=False)
ax.set_title('Iris: development validation demonstration');fig.tight_layout();save('ch04_confusion',fig)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=7)
trainmeans=[]; valmeans=[]; depths=[1,2,3,4,5,7,10,15]
for dep in depths:
    res=cross_validate(DecisionTreeClassifier(max_depth=dep,random_state=42),Xi,yi,cv=cv,return_train_score=True,scoring='accuracy')
    trainmeans.append(res['train_score'].mean());valmeans.append(res['test_score'].mean())
fig,ax=plt.subplots(figsize=(7.2,4));ax.plot(depths,trainmeans,'o-',label='Train');ax.plot(depths,valmeans,'s-',label='CV validation')
ax.set(xlabel='Maximum tree depth',ylabel='Accuracy',title='Training and cross-validation behavior');ax.legend();ax.grid(alpha=.2);fig.tight_layout();save('ch04_tree_complexity',fig)
# Chapter 5
Xscaled=StandardScaler().fit_transform(Xd)
alphas=np.logspace(-4,3,21)
coefs=np.vstack([Ridge(alpha=a).fit(Xscaled,yd).coef_ for a in alphas])
fig,ax=plt.subplots(figsize=(7.5,4.3))
for j in range(6):ax.plot(np.log10(alphas),coefs[:,j],label=f'Feature {j}')
ax.set(xlabel='log10(alpha)',ylabel='Coefficient',title='Ridge coefficient shrinkage (Diabetes demonstration)')
ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2);fig.tight_layout();save('ch05_ridge_path',fig)
# Chapter 6
xx=np.linspace(-9,9,400);fig,ax=plt.subplots(figsize=(6.8,4));ax.plot(xx,1/(1+np.exp(-xx)))
ax.axhline(.5,ls='--',color='0.5');ax.axvline(0,ls='--',color='0.5')
ax.set(xlabel='Linear score z',ylabel='Sigmoid(z)',title='From linear scores to probabilities',ylim=(-.03,1.03));fig.tight_layout();save('ch06_sigmoid',fig)
mask=yi<2;Xb=Xi[mask][:,2:4];yb=yi[mask]
Xtr,Xte,ytr,yte=train_test_split(Xb,yb,stratify=yb,test_size=.25,random_state=42)
cl=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(Xtr,ytr)
xmin,xmax=Xb[:,0].min()-.4,Xb[:,0].max()+.4; ymin,ymax=Xb[:,1].min()-.3,Xb[:,1].max()+.3
xx,yy=np.meshgrid(np.linspace(xmin,xmax,160),np.linspace(ymin,ymax,140))
Z=cl.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1].reshape(xx.shape)
fig,ax=plt.subplots(figsize=(7,4.5));ax.contour(xx,yy,Z,levels=[.2,.5,.8],linewidths=[.8,2,.8])
for k in [0,1]: ax.scatter(Xb[yb==k,0],Xb[yb==k,1],s=16,label=iris.target_names[k])
ax.set(xlabel='Petal length (cm)',ylabel='Petal width (cm)',title='Iris 2-class logistic decision boundary');ax.legend();fig.tight_layout();save('ch06_logistic_boundary',fig)
# Chapter 7
X2=Xi[:,2:4];x0,x1=X2[:,0].min()-.4,X2[:,0].max()+.4;y0,y1=X2[:,1].min()-.3,X2[:,1].max()+.3
xx,yy=np.meshgrid(np.linspace(x0,x1,120),np.linspace(y0,y1,120))
fig,axs=plt.subplots(1,2,figsize=(9.2,3.8),sharex=True,sharey=True)
for ax,k in zip(axs,[1,11]):
    kn=KNeighborsClassifier(n_neighbors=k).fit(X2,yi)
    zz=kn.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx,yy,zz,levels=np.arange(4)-.5,alpha=.22)
    ax.scatter(X2[:,0],X2[:,1],s=11,c=yi,cmap='viridis',alpha=.75)
    ax.set(title=f'k = {k}',xlabel='Petal length (cm)',ylabel='Petal width (cm)')
fig.suptitle('KNN: local to smoother boundaries');fig.tight_layout();save('ch07_knn_boundaries',fig)
ks=[1,3,5,9,15,21];raw=[];scaled=[]
for k in ks:
    raw.append(cross_val_score(KNeighborsClassifier(n_neighbors=k),Xw,yw,cv=cv).mean())
    scaled.append(cross_val_score(make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=k)),Xw,yw,cv=cv).mean())
fig,ax=plt.subplots(figsize=(7,4));ax.plot(ks,raw,'o-',label='Raw');ax.plot(ks,scaled,'s-',label='Standardized')
ax.set(xlabel='Number of neighbors (k)',ylabel='CV accuracy',title='Wine: KNN and feature scaling');ax.legend();ax.grid(alpha=.2);fig.tight_layout();save('ch07_wine_k',fig)
# Chapter 8
D=DecisionTreeClassifier(max_depth=3,min_samples_leaf=3,random_state=42).fit(Xi,yi)
fig,ax=plt.subplots(figsize=(12,5.5));plot_tree(D,max_depth=3,feature_names=iris.feature_names,class_names=list(iris.target_names),filled=True,rounded=True,fontsize=8,ax=ax)
ax.set_title('A shallow decision tree on Iris');fig.tight_layout();save('ch08_iris_tree',fig)
trainmeans=[];valmeans=[];depths=[1,2,3,4,5,7,10,15]
for dep in depths:
    r=cross_validate(DecisionTreeClassifier(max_depth=dep,random_state=42),Xw,yw,cv=cv,return_train_score=True)
    trainmeans.append(r['train_score'].mean());valmeans.append(r['test_score'].mean())
fig,ax=plt.subplots(figsize=(7,4));ax.plot(depths,trainmeans,'o-',label='Train');ax.plot(depths,valmeans,'s-',label='Validation')
ax.set(xlabel='Maximum depth',ylabel='Accuracy',title='Wine: decision tree complexity')
ax.grid(alpha=.2);ax.legend();fig.tight_layout();save('ch08_tree_depth',fig)
# Chapter 9
counts=[5,15,40,90,160];means=[];stds=[]
cv3=StratifiedKFold(n_splits=3,shuffle=True,random_state=7)
for n in counts:
    rf=RandomForestClassifier(n_estimators=n,min_samples_leaf=2,random_state=42,n_jobs=1)
    res=cross_val_score(rf,Xw,yw,cv=cv3)
    means.append(res.mean());stds.append(res.std())
fig,ax=plt.subplots(figsize=(7,4.3));ax.errorbar(counts,means,yerr=stds,marker='o',capsize=4)
ax.set(xlabel='Number of trees',ylabel='3-fold CV accuracy',title='Wine: forest size and validation variability')
ax.grid(alpha=.2);fig.tight_layout();save('ch09_rf_trees',fig)
rf=RandomForestClassifier(n_estimators=180,min_samples_leaf=2,random_state=42,n_jobs=1).fit(Xw,yw)
order=np.argsort(rf.feature_importances_)[-10:]
fig,ax=plt.subplots(figsize=(8.4,5.1));ax.barh(np.array(wine.feature_names)[order],rf.feature_importances_[order])
ax.set(xlabel='Impurity-based feature importance',title='Wine: illustrative model feature dependence')
fig.tight_layout();save('ch09_rf_importance',fig)
# Chapter 10
rng=np.random.default_rng(42)
x=np.linspace(-3.0,3.0,90)
y=np.sin(1.5*x)+rng.normal(0,.14,len(x))
boost=GradientBoostingRegressor(n_estimators=70,learning_rate=.12,max_depth=2,random_state=42).fit(x[:,None],y)
xx=np.linspace(-3.3,3.3,380)
scored={1:None,10:None,70:None}
for idx,pred in enumerate(boost.staged_predict(xx[:,None]),1):
    if idx in scored:scored[idx]=pred
fig,ax=plt.subplots(figsize=(8.2,4.2));ax.scatter(x,y,s=12,label='Synthetic observations',alpha=.65)
for n in scored:ax.plot(xx,scored[n],label=f'After {n} trees')
ax.set(xlabel='Synthetic x',ylabel='Synthetic target',title='Boosting: sequential approximation')
ax.legend(ncol=2,fontsize=8);fig.tight_layout();save('ch10_boosting_steps',fig)
# Small built-in Diabetes regression for purely methodological comparison
cvsmall=KFold(n_splits=3,shuffle=True,random_state=7)
counts=[10,30,60,100];rf_scores=[];gbr_scores=[]
for n in counts:
    rf=RandomForestRegressor(n_estimators=n,max_depth=4,random_state=42,n_jobs=1)
    gb=GradientBoostingRegressor(n_estimators=n,learning_rate=.08,max_depth=2,random_state=42)
    rf_scores.append(-cross_val_score(rf,Xd,yd,cv=cvsmall,scoring='neg_mean_absolute_error').mean())
    gbr_scores.append(-cross_val_score(gb,Xd,yd,cv=cvsmall,scoring='neg_mean_absolute_error').mean())
fig,ax=plt.subplots(figsize=(7.3,4));ax.plot(counts,rf_scores,'o-',label='Random forest');ax.plot(counts,gbr_scores,'s-',label='Gradient boosting')
ax.set(xlabel='Number of trees',ylabel='3-fold CV MAE',title='Diabetes: illustration of ensemble comparison')
ax.legend();ax.grid(alpha=.2);fig.tight_layout();save('ch10_ensemble_compare',fig)
# Chapter 11
Xm,ym=make_blobs(n_samples=120,centers=[[-1.5,0],[1.5,0]],cluster_std=.55,random_state=42)
svm=SVC(kernel='linear',C=10).fit(Xm,ym)
fig,ax=plt.subplots(figsize=(7,4.2));ax.scatter(Xm[:,0],Xm[:,1],c=ym,cmap='coolwarm',s=17)
Xg,Yg=np.meshgrid(np.linspace(-4,4,150),np.linspace(-2.6,2.6,150))
Z=svm.decision_function(np.c_[Xg.ravel(),Yg.ravel()]).reshape(Xg.shape)
ax.contour(Xg,Yg,Z,levels=[-1,0,1],linestyles=['--','-','--'],colors='k',linewidths=[.8,1.7,.8])
ax.scatter(svm.support_vectors_[:,0],svm.support_vectors_[:,1],s=90,facecolors='none',edgecolors='k',label='Support vectors')
ax.set(xlabel='Feature 1',ylabel='Feature 2',title='Linear SVM: margin and support vectors')
ax.legend(loc='upper left');fig.tight_layout();save('ch11_svm_margin',fig)
Xm,ym=make_moons(n_samples=300,noise=.16,random_state=42)
Xg,Yg=np.meshgrid(np.linspace(-1.8,2.7,130),np.linspace(-1.2,1.8,130))
fig,axs=plt.subplots(1,2,figsize=(9.1,3.8),sharex=True,sharey=True)
for ax,kernel in zip(axs,['linear','rbf']):
    clf=SVC(kernel=kernel,C=2,gamma=2).fit(Xm,ym)
    Z=clf.decision_function(np.c_[Xg.ravel(),Yg.ravel()]).reshape(Xg.shape)
    ax.contour(Xg,Yg,Z,levels=[0],colors='k',linewidths=1.6)
    ax.scatter(Xm[:,0],Xm[:,1],c=ym,cmap='coolwarm',s=8,alpha=.8)
    ax.set(title=f'SVC {kernel}',xlabel='Feature 1',ylabel='Feature 2')
fig.tight_layout();save('ch11_svm_kernels',fig)
# Chapter 12
fig,axs=plt.subplots(1,2,figsize=(10.5,4.6))
for ax,layers,cap in [(axs[0],[3,1],'Linear score + sigmoid'),(axs[1],[3,4,1],'MLP with hidden layer')]:
    ax.set_xlim(-.4,len(layers)-.6);ax.set_ylim(-.25,4.25);ax.axis('off')
    positions=[]
    for li,n in enumerate(layers):
        ys=np.linspace(2.1-(n-1)*.45,2.1+(n-1)*.45,n)
        positions.append([(li,float(y)) for y in ys])
    for cur,nxt in zip(positions[:-1],positions[1:]):
        for x1,y1 in cur:
            for x2,y2 in nxt:ax.plot([x1,x2],[y1,y2],lw=.85,alpha=.4,color='0.45')
    for li,layer in enumerate(positions):
        for x,y in layer:
            ax.scatter(x,y,s=460,color='white',edgecolors='black',zorder=5)
            ax.text(x,y,('x' if li==0 else ('p' if li==len(positions)-1 else 'h')),ha='center',va='center',zorder=6)
    ax.set_title(cap)
fig.tight_layout();save('ch12_network',fig)
# Wine MLP on fixed train split (training loss ONLY)
Xtr,Xva,ytr,yva=train_test_split(Xw,yw,stratify=yw,test_size=.25,random_state=42)
scaler=StandardScaler().fit(Xtr);XX=scaler.transform(Xtr)
net=MLPClassifier(hidden_layer_sizes=(20,),alpha=.01,solver='adam',max_iter=650,random_state=42).fit(XX,ytr)
fig,ax=plt.subplots(figsize=(7,4.0));ax.plot(np.arange(1,len(net.loss_curve_)+1),net.loss_curve_)
ax.set(xlabel='Training iterations',ylabel='Training loss',title='Wine MLP: training objective trajectory')
ax.grid(alpha=.2);fig.tight_layout();save('ch12_mlp_loss',fig)
fig,axs=plt.subplots(2,5,figsize=(9.1,4.3))
for ax,n in zip(axs.flat,range(10)):
    ax.imshow(digits.images[n],cmap='gray_r',interpolation='nearest');ax.set_title(f'Label {digits.target[n]}');ax.axis('off')
fig.suptitle('Digits: real 8x8 example images bundled with sklearn');fig.tight_layout();save('ch12_digits',fig)
# Chapter 13
Xstd=StandardScaler().fit_transform(Xw)
pca=PCA(n_components=2).fit(Xstd)
Xp=pca.transform(Xstd)
fig,ax=plt.subplots(figsize=(7.3,4.5))
for k in range(3):ax.scatter(Xp[yw==k,0],Xp[yw==k,1],s=18,alpha=.75,label=f'Class {k}')
ax.set(xlabel=f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)',ylabel=f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)',title='Wine PCA (classes used for display only)')
ax.legend();fig.tight_layout();save('ch13_wine_pca',fig)
pca=PCA(n_components=12).fit(digits.data)
inds=np.array([0,5,10,20,30,40]);rec=pca.inverse_transform(pca.transform(digits.data[inds]))
fig,axs=plt.subplots(2,6,figsize=(10.1,3.6))
for i in range(6):
    axs[0,i].imshow(digits.data[inds[i]].reshape(8,8),cmap='gray_r',vmin=0,vmax=16)
    axs[1,i].imshow(rec[i].reshape(8,8),cmap='gray_r',vmin=0,vmax=16)
    for ax in axs[:,i]:ax.axis('off')
axs[0,0].set_ylabel('Original',rotation=90);axs[1,0].set_ylabel('PCA-12',rotation=90)
fig.suptitle('Digits reconstruction from 12 principal components');fig.tight_layout();save('ch13_digits_recon',fig)
# Chapter 14
Xm,ym=make_moons(n_samples=360,noise=.08,random_state=42)
Xm=StandardScaler().fit_transform(Xm)
yk=KMeans(n_clusters=2,n_init=10,random_state=42).fit_predict(Xm)
ydb=DBSCAN(eps=.24,min_samples=5).fit_predict(Xm)
fig,axs=plt.subplots(1,2,figsize=(9.2,3.8),sharex=True,sharey=True)
for ax,yy,t in zip(axs,[yk,ydb],['K-means: 2 clusters','DBSCAN: density groups']):
    ax.scatter(Xm[:,0],Xm[:,1],c=yy,s=13,cmap='viridis');ax.set(title=t,xlabel='Scaled feature 1',ylabel='Scaled feature 2')
fig.tight_layout();save('ch14_cluster_shapes',fig)
Xb,yb=make_blobs(n_samples=450,centers=4,cluster_std=1.1,random_state=42)
inertia=[]
for k in range(1,11):inertia.append(KMeans(n_clusters=k,n_init=10,random_state=42).fit(Xb).inertia_)
fig,ax=plt.subplots(figsize=(7,4));ax.plot(range(1,11),inertia,'o-')
ax.set(xlabel='Number of clusters (k)',ylabel='Within-cluster sum of squares',title='Synthetic blobs: K-means objective vs k')
ax.grid(alpha=.2);fig.tight_layout();save('ch14_kmeans_k',fig)
# Chapter 15
rng=np.random.default_rng(42); normal=rng.normal(size=(240,2));injected=rng.uniform(-6,6,size=(15,2))
Xout=np.vstack((normal,injected)); truth=np.r_[np.zeros(len(normal),int),np.ones(len(injected),int)]
outmodel=IsolationForest(n_estimators=150,contamination='auto',random_state=42).fit(Xout)
score=-outmodel.score_samples(Xout)
fig,ax=plt.subplots(figsize=(7,4.6)); scat=ax.scatter(Xout[:,0],Xout[:,1],c=score,s=21,cmap='viridis')
ax.scatter(injected[:,0],injected[:,1],s=70,facecolors='none',edgecolors='black',linewidths=1.2,label='Injected points')
ax.set(xlabel='Synthetic feature 1',ylabel='Synthetic feature 2',title='IsolationForest: synthetic outlier scores');ax.legend()
fig.colorbar(scat,ax=ax,label='-score_samples (higher = more unusual)');fig.tight_layout();save('ch15_isolation',fig)
quantiles=np.linspace(.75,.99,13);ths=np.quantile(score,quantiles)
counts=[];hits=[]
for t in ths:
    sel=score>=t
    counts.append(int(sel.sum()));hits.append(int(np.sum(sel & (truth==1))))
fig,ax=plt.subplots(figsize=(7.5,4));ax.plot(quantiles,counts,'o-',label='All candidates');ax.plot(quantiles,hits,'s-',label='Detected injected points')
ax.set(xlabel='Outlier-score quantile threshold',ylabel='Count',title='Changing the threshold changes candidate yield')
ax.legend();ax.grid(alpha=.2);fig.tight_layout();save('ch15_threshold',fig)
# Chapter 16
Xtr,Xval,ytr,yval=train_test_split(Xw,yw,stratify=yw,test_size=.30,random_state=42)
model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(Xtr,ytr)
fig,ax=plt.subplots(figsize=(5.3,4.5));ConfusionMatrixDisplay.from_estimator(model,Xval,yval,ax=ax,cmap='Blues',colorbar=False)
ax.set_title('Wine: validation errors');fig.tight_layout();save('ch16_wine_confusion',fig)
p=permutation_importance(model,Xval,yval,n_repeats=10,random_state=42,scoring='accuracy')
order=np.argsort(p.importances_mean)[-9:]
fig,ax=plt.subplots(figsize=(8.5,4.8));ax.barh(np.array(wine.feature_names)[order],p.importances_mean[order],xerr=p.importances_std[order])
ax.set(xlabel='Decrease in validation accuracy',title='Wine: permutation importance (10 repeats)');fig.tight_layout();save('ch16_permutation',fig)
Xtr,Xval,ytr,yval=train_test_split(Xd,yd,test_size=.3,random_state=42)
reg=Ridge(alpha=1.0).fit(Xtr,ytr)
residual=reg.predict(Xval)-yval
cuts=np.quantile(yval,[0,.25,.5,.75,1])
fig,ax=plt.subplots(figsize=(8,4.4))
inds=np.digitize(yval,cuts[1:-1]);groups=[residual[inds==k] for k in range(4)]
ax.boxplot(groups,tick_labels=['Q1','Q2','Q3','Q4'],showmeans=True)
ax.axhline(0,color='0.5',ls='--')
ax.set(xlabel='Target quartile (validation records)',ylabel='Prediction - target',title='Diabetes: grouped regression residual illustration')
fig.tight_layout();save('ch16_group_residual',fig)
# Chapter 17
workflow('ch17_project_flow',['Research question','Data audit','Split design','Baseline','Model + CV','Final evaluation','Interpretation'])
print(f'Generated {len(GENERATED)} figure groups, {len(GENERATED)*2} PDF/PNG files in {DEST}')
for s in GENERATED:print(s)
