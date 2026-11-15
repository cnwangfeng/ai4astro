"""Chapter 11: margins, kernels and a controlled C/gamma experiment."""
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from common import *

def run():
    X,y,_=development('wine');splits=list(cv5().split(X,y));rows=[]
    for C in [.1,1.,10.]:
        for gamma in [.01,.1,1.]:
            model=make_pipeline(StandardScaler(),SVC(C=C,gamma=gamma,kernel='rbf'))
            scores=cross_val_score(model,X,y,cv=splits)
            rows.append({'C':C,'gamma':gamma,'validation_accuracy':scores.mean(),'fold_std':scores.std()})
    df=table('ch11_grid',rows);show_result(df)
    plt.figure(figsize=(6.4,4))
    for C in [.1,1.,10.]:
        d=df[df.C==C];plt.semilogx(d.gamma,d.validation_accuracy,'o-',label=f'C={C}')
    plt.xlabel('RBF gamma');plt.ylabel('五折验证准确率');plt.legend();figure('v3_ch11_grid')
    # A hand-constructed nonlinear problem, with no held-out scientific interpretation.
    xx=np.array([[-2,0],[0,-2],[2,0],[0,2],[.2,0],[0,.2],[-.2,0],[0,-.2]])
    yy=np.array([1,1,1,1,0,0,0,0])
    lin=SVC(kernel='linear',C=100).fit(xx,yy);rbf=SVC(kernel='rbf',C=100,gamma=1).fit(xx,yy)
    dump('ch11_nonlinear',{'linear_training_accuracy':lin.score(xx,yy),'rbf_training_accuracy':rbf.score(xx,yy)})
    a=np.array([1.,2.]);b=np.array([3.,-1.]);phi=lambda v:np.array([v[0]**2,np.sqrt(2)*v[0]*v[1],v[1]**2])
    assert np.isclose(phi(a)@phi(b),(a@b)**2)
    table('ch11_kernel_check',[{'explicit_feature_dot':phi(a)@phi(b),'squared_original_dot':(a@b)**2}])
if __name__=='__main__':run()
