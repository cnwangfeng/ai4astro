"""Chapter 1: a complete task with interpretable baseline and an explicit table."""
import numpy as np
import pandas as pd
from common import *

def run():
    Xtr,Xva,ytr,yva,ds=development_trainval('iris')
    # This is a deliberately prescribed rule, not an optimized decision tree.
    positive=(yva==0).to_numpy()
    pred=(Xva['petal length (cm)'].to_numpy()<2.5)
    majority=np.full(len(positive),int(np.mean(ytr==0)>=.5))
    df=table('ch01_task',[{'rule':'majority baseline','n_validation':len(yva),'correct':int(np.sum(majority==positive))},
        {'rule':'prescribed petal threshold','n_validation':len(yva),'correct':int(np.sum(pred==positive))}])
    preview=Xva.iloc[:6].copy();preview['reference_is_setosa']=positive[:6];preview['predicted_is_setosa']=pred[:6]
    table('ch01_records',preview.reset_index(names='original_row'))
    show_result(df)
if __name__=='__main__':run()
