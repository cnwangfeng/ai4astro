#!/usr/bin/env python3
"""Independent small calculations and cross-path checks for textbook examples.
Does not import the per-chapter experiment functions.
"""
from pathlib import Path
import json,math
import numpy as np,pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import LocalOutlierFactor
R=Path(__file__).resolve().parents[1];checks=[]
def record(name,actual,expected,atol=1e-9,rtol=1e-8):
    a=np.asarray(actual);e=np.asarray(expected)
    ok=bool(np.allclose(a,e,atol=atol,rtol=rtol,equal_nan=True))
    checks.append({'name':name,'passed':ok,'actual':a.tolist(),'expected':e.tolist()})
    if not ok:raise AssertionError(name+': '+str(actual)+' != '+str(expected))
def gini(y):
    p=np.unique(y,return_counts=True)[1]/len(y);return 1-float(p@p)
def entropy(y):
    p=np.unique(y,return_counts=True)[1]/len(y);return float(-(p*np.log2(p)).sum())
# 1–3: scale, loss and full gradients.
a=np.array([1.,3.,5.]);record('training_standardization',(7-a.mean())/a.std(),np.sqrt(6))
x=np.array([1.,2.,3.]);y=np.array([3.,5.,8.]);D=np.column_stack([x,np.ones(3)]);theta=np.array([1.,0.])
f=lambda z:np.mean((D@z-y)**2);grad=2/3*D.T@(D@theta-y)
record('full_MSE_gradient',grad,[-46/3,-20/3])
eps=1e-6;gn=np.array([(f(theta+eps*e)-f(theta-eps*e))/(2*eps) for e in np.eye(2)])
record('linear_gradient_finite_difference',gn,grad,atol=1e-8)
H=2/3*D.T@D;record('quadratic_Hessian',H,[[28/3,4],[4,2]])
record('anisotropic_one_step',np.array([1.,1.])-.2*np.array([4.,1.]),[.2,.8])
record('anisotropic_target_after',.5*(4*.2**2+.8**2),.4)
# Evaluation and OLS.
record('rare_target_precision',90/(90+495),2/13)
y=np.array([2.,3.,5.]);beta=np.linalg.lstsq(np.column_stack([np.ones(3),x]),y,rcond=None)[0]
record('three_point_OLS_coefficients',beta,[1/3,1.5])
res=np.column_stack([np.ones(3),x])@beta-y;record('OLS_SSE',res@res,1/6);record('OLS_residual_sum',res.sum(),0)
lam=np.array([0,.5,1,2,3]);record('L1_soft_threshold',np.maximum(2-lam,0),[2,1.5,1,0,0]);record('L2_shrinkage',2/(1+lam),[2,4/3,1,2/3,.5])
# Logistic and thresholds, independently recomputed.
x=np.array([-1.,0.,1.]);y=np.array([0.,0.,1.]);p=np.full(3,.5)
gw=np.mean((p-y)*x);gb=np.mean(p-y);record('logistic_initial_gradient',[gw,gb],[-1/3,1/6])
z=(-.3*gw)*x-.3*gb;newloss=np.mean(np.logaddexp(0,z)-y*z);record('logistic_updated_loss',newloss,.652625447872366,atol=2e-10)
probs=np.array([.95,.8,.6,.55,.4,.3,.1,.05]);labels=np.array([1,1,0,1,0,1,0,0])
for t,counts in [(.2,[4,2,0,2]),(.5,[3,1,1,3]),(.8,[2,0,2,4])]:
 pred=probs>=t;actual=[sum(pred&(labels==1)),sum(pred&(labels==0)),sum(~pred&(labels==1)),sum(~pred&(labels==0))];record(f'threshold_{t}',actual,counts)
record('cost_threshold',1/(1+9),.1)
logits=np.log([2.,3.,1.]);sm=lambda a:np.exp(a-a.max())/np.exp(a-a.max()).sum();record('softmax_shift',sm(logits+1000),[1/3,1/2,1/6])
# Neighbors.
record('distance_weighted_regression',(2/1+8/2)/(1/1+1/2),4)
record('Gaussian_squared_distance_CV_p50',math.sqrt(8*50)/(2*50),math.sqrt(2/50))
# All thresholds for the uploaded benchmark's corrected example.
x=np.array([1,1.6,2,2.4,2.8,3,3.4,3.6,4.2,5.]);y=np.array([0,0,0,0,0,1,0,1,1,1])
thresholds=(x[:-1]+x[1:])/2;gains=[]
for t in thresholds:
 left=x<=t;gains.append(gini(y)-left.mean()*gini(y[left])-(~left).mean()*gini(y[~left]))
record('classification_best_threshold',thresholds[np.argmax(gains)],2.9)
record('classification_best_gain',max(gains),.32)
record('comparison_2p6_3p2_3p5',[gains[3],gains[5],gains[6]],[16/75,49/300,54/175])
record('node_entropy_5_3',entropy([0]*5+[1]*3),.954434002924965)
record('weighted_Gini',1-.2**2-.8**2,.32)
x=np.arange(1.,7.);y=np.array([1,2,3,7,8,9.]);sse=lambda a:np.sum((a-a.mean())**2)
record('regression_parent_SSE',sse(y),58);record('regression_child_SSE',sse(y[:3])+sse(y[3:]),4)
record('regression_other_threshold_SSE',sse(y[:1])+sse(y[1:]),38.8)
record('whole_tree_impurity_decrease',20/100*.1,.02)
X=np.array([(a,b) for a in (1,2,3) for b in (1,2,3)]);y=np.array([0,0,0,0,0,1,1,1,1])
m=DecisionTreeClassifier(max_depth=2,random_state=42).fit(X,y)
record('greedy_grid_errors',sum(m.predict(X)!=y),1)
alt=np.where(X[:,1]<=2.5,np.where(X[:,0]<=2.5,0,1),np.where(X[:,0]<=1.5,0,1));record('alternative_grid_errors',sum(alt!=y),0)
for alpha,chosen in [(.15,6),(.3,2),(12,1)]:
 leaves=np.array([1,2,6]);obj=np.array([29/3,2/3,0])+alpha*leaves;record(f'pruning_candidate_alpha_{alpha}',leaves[obj.argmin()],chosen)
# Ensembles and sequential updates.
record('average_error_variance_B10_rho_p2',(.2+(1-.2)/10),.28)
record('bootstrap_oob_n106',(1-1/106)**106,.3661373,atol=1e-6)
record('soft_versus_hard_vote',np.mean([.51,.51,.01]),1.03/3)
y=np.array([0.,1.,2.,4.]);F=np.full(4,y.mean());h=np.array([-.75,-.75,-.75,2.25]);F=F+.5*h
record('boost_first_prediction',F,[1.375,1.375,1.375,2.875]);record('boost_first_MSE',np.mean((F-y)**2),.921875)
r=y-F;h2=np.array([r[:2].mean()]*2+[r[2:].mean()]*2);F2=F+.5*h2
record('boost_second_prediction',F2,[.9375,.9375,1.8125,3.3125]);record('boost_second_MSE',np.mean((F2-y)**2),.34765625)
record('logistic_negative_gradient_at_zero',1-.5,.5)
# Kernel and network.
record('hinge_loss',np.maximum(0,1-np.array([1.5,.4,-.3])),[0,.6,1.3])
phi=lambda x:np.array([x[0]**2,np.sqrt(2)*x[0]*x[1],x[1]**2]);record('polynomial_kernel',phi([1,2])@phi([3,4]),121)
XX=np.array([[0,0],[0,1],[1,0],[1,1]]);record('XOR_ReLU',np.maximum(XX[:,0]-XX[:,1],0)+np.maximum(XX[:,1]-XX[:,0],0),[0,1,1,0])
record('MLP_parameter_counts',[13*8+8+8*3+3,13*8+8+8*4+4+4*3+3],[139,163])
t=np.array([.5,0.,3.,0.]);netloss=lambda t:.5*(t[2]*max(0,t[0]*2+t[1])+t[3]-1)**2
g=np.array([(netloss(t+eps*e)-netloss(t-eps*e))/(2*eps) for e in np.eye(4)]);record('backprop_numerical_gradient',g,[12,6,2,2],atol=1e-8)
record('network_loss_after',netloss(t-.01*np.array([12,6,2,2])),.568178)
# PCA and clustering.
X=np.array([[2.,1.],[1.,2.],[-1.,-2.],[-2.,-1.]]);S=X.T@X/3
record('PCA_covariance',S,[[10/3,8/3],[8/3,10/3]]);record('PCA_eigenvalues',np.linalg.eigvalsh(S),[2/3,6])
v=np.array([1.,1.])/np.sqrt(2);hat=np.outer(X@v,v);record('PCA_reconstruction_SSE',np.sum((X-hat)**2),2)
x=np.array([0.,1.,4.,5.,9.,10.]);record('Lloyd_local_SSE',sse(x[:2])+sse(x[2:]),26.5);record('alternative_cluster_SSE',sse(x[:4])+sse(x[4:]),17.5)
lof=LocalOutlierFactor(n_neighbors=2).fit(np.array([0.,1.,2.,10.]).reshape(-1,1));record('LOF_hand_vs_library',-lof.negative_outlier_factor_[-1],119/24,atol=2e-8)
record('conformal_finite_rank',math.ceil((19+1)*.9),18)
# Gaussian process numeric example in Appendix E: k(x,x')=exp(-(x-x')^2/2), noise variance .01.
x=np.array([0.,1.]);k=lambda a,b:np.exp(-.5*(a-b)**2);K=k(x[:,None],x[None,:])+.01*np.eye(2);ks=k(.5,x)
mu=ks@np.linalg.solve(K,np.array([0.,1.]));var=1-ks@np.linalg.solve(K,ks)
record('GP_two_point_mean',mu,.5459203026779713);record('GP_two_point_latent_variance',var,.036454052520290325)
# Scientific project: independent numpy fit and held-out metric, not experiment import.
df=pd.read_csv(R/'data/ogle_lmc_f_excerpt.csv',dtype={'source_id':'string'});record('OGLE_raw_rows',len(df),90);record('OGLE_unique_IDs',df.source_id.nunique(),90)
valid=df.I_mag.notna() & (df.period_days>0);record('OGLE_valid_rows',valid.sum(),87)
split=pd.read_csv(R/'data/ogle_excerpt_split_v3.csv');merged=df.merge(split,on=['source_row','source_id'],how='inner',validate='one_to_one')
tr=merged[merged['split']=='development'];te=merged[merged['split']=='test'];record('OGLE_role_counts',[len(tr),len(te)],[69,18])
A=np.column_stack([np.log10(tr.period_days),np.ones(len(tr))]);beta=np.linalg.lstsq(A,tr.I_mag,rcond=None)[0]
record('OGLE_numpy_slope_intercept',beta,[-2.8805611804106284,16.902095076747])
pred=np.column_stack([np.log10(te.period_days),np.ones(len(te))])@beta
record('OGLE_numpy_test_RMSE',np.sqrt(np.mean((pred-te.I_mag)**2)),.14919339587341696)
s=pd.read_csv(R/'results/ch17_splits.csv');dev=set(s.loc[s.role=='development','run_id']);hold=set(s.loc[s.role=='test','run_id'])
record('calibration_disjoint_groups',len(dev&hold),0);record('calibration_group_counts',[len(dev),len(hold)],[72,18])
out={'checks':checks,'passed':sum(c['passed'] for c in checks),'total':len(checks),'scope':'Independent small calculations and alternate numerical paths; not an independent academic reviewer.'}
(R/'review_v3/numerical_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(out['passed'],'/',out['total'],'numerical checks passed')
