"""Chapter 3: gradients, curvature and feature-scale stability (additional checks)."""
import numpy as np
from common import *

def run():
    x=np.array([1.,2.,3.]); y=np.array([3.,5.,8.]); D=np.column_stack([x,np.ones(3)])
    theta=np.array([1.,0.]);loss=lambda t:np.mean((D@t-y)**2)
    g=2*D.T@(D@theta-y)/len(y)
    eps=1e-6
    numerical=np.array([(loss(theta+eps*np.eye(2)[j])-loss(theta-eps*np.eye(2)[j]))/(2*eps) for j in range(2)])
    assert np.allclose(g,numerical,rtol=1e-7)
    H=2*D.T@D/len(y);eig=np.linalg.eigvalsh(H)
    df=table('ch03_gradient',{'component':['slope','intercept'],'analytic':g,'finite_difference':numerical})
    dump('ch03_curvature',{'hessian':H,'eigenvalues':eig,'constant_step_stability_upper':2/eig[-1]})
    show_result(df)
    rows=[]
    for eta in [.1,.5,.95,1.,1.05]:
        c=0.
        for step in range(21):
            rows.append({'eta':eta,'step':step,'parameter':c,'loss':(c-2)**2})
            c=c-eta*2*(c-2)
    table('ch03_scalar_trace',rows)
    plt.figure(figsize=(6.4,4))
    for eta in [.1,.5,.95,1.05]:
        d=pd.DataFrame(rows);d=d[d.eta==eta]
        plt.plot(d.step,d.parameter,marker='.',label=f'eta={eta}')
    plt.axhline(2,linestyle='--');plt.xlabel('更新次数');plt.ylabel('参数 c');plt.legend();figure('v3_ch03_steps')
if __name__=='__main__':run()
