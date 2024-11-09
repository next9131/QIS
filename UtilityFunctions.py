import numpy as np

def indices(a, func):
    #This is like find in Matlab
    return [i for (i, val) in enumerate(a) if func(val)]

def intersect(a, b):
    return list(set(a) & set(b))

def isempty(x):
    ie =    (x==[])*1
    return ie

def removenan(Data):
    CData       =   Data.copy()
    NotNanInds  =   indices(CData, lambda x: not np.isnan(x))#find not nan positions
    NotNanVals  =   CData[NotNanInds]
    return NotNanVals,NotNanInds

def rem(a,b):
    return a%b

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return ys

def pairwise(a,b):
    Data        =   [a,b]
    Data        =   np.sum(Data,axis=0)
    goodlist    =   indices(Data, lambda x: not np.isnan(x))#find not nan positions
    if len(a.shape) == 1:
        a =   a[goodlist]
    elif len(a.shape)==2:
        a = a[:,goodlist]
        
    if len(b.shape) == 1:
        b =   b[goodlist]
    elif len(b.shape)==2:
        b = b[:,goodlist]
    return a,b,goodlist