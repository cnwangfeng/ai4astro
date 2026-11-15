"""Chapter 5: complete least squares and polynomial complexity examples."""
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error,mean_absolute_error
from common import *

def run():
    x=np.array([1.,2.,3.]);y=np.array([2.,3.,5.]);D=np.column_stack([np.ones(3),x])
    beta=np.linalg.lstsq(D,y,rcond=None)[0]
    assert np.allclose(beta,[1/3,1.5])
    assert np.allclose(D.T@(D@beta-y),0,atol=1e-12)
    table('ch05_hand',{'x':x,'y':y,'prediction':D@beta,'residual_y_minus_pred':y-D@beta})
    tr,va,_=regression_demo();rows=[]
    for degree in [1,2,3,5,9,15]:
        m=make_pipeline(PolynomialFeatures(degree,include_bias=False),StandardScaler(),Ridge(alpha=1e-6))
        m.fit(tr[['x']],tr.target)
        rows.append({'degree':degree,'train_RMSE':np.sqrt(mean_squared_error(tr.target,m.predict(tr[['x']]))),
                     'validation_RMSE':np.sqrt(mean_squared_error(va.target,m.predict(va[['x']])) )})
    df=table('ch05_polynomial',rows);show_result(df)
    plt.figure(figsize=(6.4,4));plt.plot(df.degree,df.train_RMSE,'o-',label='training');plt.plot(df.degree,df.validation_RMSE,'s-',label='validation')
    plt.xlabel('多项式次数');plt.ylabel('RMSE（人工单位）');plt.legend();figure('v3_ch05_degree')
    # Collinear features: coefficient interpretation is not prediction performance.
    rng=np.random.default_rng(5);a=rng.normal(size=100);Z=np.column_stack([a,a+rng.normal(0,.01,100)])
    y2=2*a+rng.normal(0,.2,100);rows=[]
    for seed in range(12):
        ids=np.random.default_rng(seed).choice(len(a),len(a),replace=True)
        for name,m in [('OLS',LinearRegression()),('Ridge',Ridge(alpha=1.))]:
            m.fit(Z[ids],y2[ids]);rows.append({'method':name,'seed':seed,'b1':m.coef_[0],'b2':m.coef_[1],'sum':m.coef_.sum()})
    df=table('ch05_collinearity',rows)
    plt.figure(figsize=(6.4,4))
    for name in ['OLS','Ridge']:
        s=df[df.method==name];plt.scatter(s.b1,s.b2,label=name)
    plt.xlabel('第一个系数');plt.ylabel('第二个系数');plt.legend();figure('v3_ch05_collinearity')
    # One-dimensional normalized objective: J=(w-2)^2/2 + penalty.
    lam=np.array([0,.5,1,2,3]);out=table('ch05_shrinkage',{'lambda':lam,'ridge':2/(1+lam),'lasso':np.maximum(2-lam,0)})
    show_result(out)
if __name__=='__main__':run()
