"""Chapter 13: derive PCA directions, reconstruction and downstream comparisons."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from common import *

def run():
    X=np.array([[2.,1.],[1.,2.],[-1.,-2.],[-2.,-1.]])
    S=np.cov(X,rowvar=False);vals,vec=np.linalg.eigh(S);v=vec[:,-1]
    if v[0]<0:v=-v
    scores=X@v;recon=np.outer(scores,v)
    assert np.allclose(vals,[2/3,6]) and np.isclose(np.mean(np.sum((X-recon)**2,axis=1)),.5)
    table('ch13_hand',{'x1':X[:,0],'x2':X[:,1],'score':scores,'recon_x1':recon[:,0],'recon_x2':recon[:,1],'squared_distance':np.sum((X-recon)**2,axis=1)})
    dump('ch13_covariance',{'covariance':S,'eigenvalues_ascending':vals,'first_direction':v,'first_variance_ratio':.9})
    plt.figure(figsize=(5.6,5));plt.scatter(X[:,0],X[:,1],label='original');plt.scatter(recon[:,0],recon[:,1],marker='x',label='rank-one reconstruction')
    for a,b in zip(X,recon):plt.plot([a[0],b[0]],[a[1],b[1]],linestyle='--')
    z=np.linspace(-2.5,2.5,50);plt.plot(z,z);plt.xlabel('x1');plt.ylabel('x2');plt.axis('equal');plt.legend();figure('v3_ch13_geometry')
    X,y,_=development('wine');splits=list(cv5().split(X,y));rows=[]
    for k in [1,2,3,5,8,13]:
        m=make_pipeline(StandardScaler(),PCA(n_components=k,svd_solver='full'),LogisticRegression(max_iter=3000))
        r=cross_val_score(m,X,y,cv=splits);rows.append({'components':k,'accuracy_mean':r.mean(),'accuracy_std':r.std()})
    df=table('ch13_downstream',rows);show_result(df)
    Z=StandardScaler().fit_transform(X);p=PCA(svd_solver='full').fit(Z)
    table('ch13_variance',{'component':np.arange(1,14),'variance_ratio':p.explained_variance_ratio_,'cumulative':np.cumsum(p.explained_variance_ratio_)})
    # Reconstruction uses only training-fitted transform; this digit is held-out within development.
    Dt,Dv,yt,yv,_=development_trainval('digits');records=[]
    for k in [2,8,20,40]:
        p=PCA(k,svd_solver='full').fit(Dt);hat=p.inverse_transform(p.transform(Dv));mse=np.mean((Dv.to_numpy()-hat)**2)
        records.append({'components':k,'validation_pixel_MSE':mse})
    table('ch13_digits_reconstruction',records)
    plt.figure(figsize=(6.4,4));plt.plot(df.components,df.accuracy_mean,'o-');plt.xlabel('保留主成分数');plt.ylabel('五折验证准确率');figure('v3_ch13_components')
if __name__=='__main__':run()
