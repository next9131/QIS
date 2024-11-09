import numpy            as np
import math
import UtilityFunctions as ut
import scipy as sp
import scipy.stats as stats
#import matplotlib.pyplot as plt



def beta(a,b): 
    '''uses gamma function or inbuilt math.gamma() to compute values of beta function'''  
    beta = math.gamma(a)*math.gamma(b)/math.gamma(a+b)
    return beta

def EstimateGLD_ML(X,GLDRegion):
    # Estimate GLD using: 
    # TECHNOMETRICS, FEBRUARY 1985, VOL. 27, NO. 1
    # Least Squares Estimation of the Parameters of the Generalized Lambda Distribution
    # by Aydin Ozturk and Robert F. Dale
    #fprime=None, args=(X), approx_grad=True,
    RegionBounds    =   setregionbounds(GLDRegion)
    initvalues      =   setinitvaluesL1L2L3L4()
    print(' Region bounds = ' + str(RegionBounds))
    print('Initial values = ' + str(initvalues))
    OptParameters = sp.optimize.least_squares(LikelihoodGLD, x0=initvalues, 
          args=(X), verbose=1)
#    R = LikelihoodGLD2(OptParameters.x,X)
#    print(OptL3L4)
    return OptParameters

def EstimateGLD_LS(X,GLDRegion):
    # Estimate GLD using: 
    # TECHNOMETRICS, FEBRUARY 1985, VOL. 27, NO. 1
    # Least Squares Estimation of the Parameters of the Generalized Lambda Distribution
    # by Aydin Ozturk and Robert F. Dale
    #fprime=None, args=(X), approx_grad=True,
#    SaveFile    = 'Xsaved.pkl'
#    file        = open(SaveFile, 'wb')
#    pk.dump(X,file)
#    file.close()
    RegionBounds    =   setregionbounds(GLDRegion)
    initvalues      =   setinitvalues(RegionBounds)
    print(' Region bounds = ' + str(RegionBounds))
    print('Initial values = ' + str(initvalues))
#    OptL3L4,fval = sp.optimize.fmin_l_bfgs_b(RegressionEstimateL3L4_MaxRsq, x0 = initvalues, 
#    fprime = None, args=(X), approx_grad = False,
#    bounds = RegionBounds, 
#    m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=-1, 
#    maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)

#    OptL3L4,fval =  sp.optimize.fmin_slsqp(RegressionEstimateL3L4_MaxRsq, x0 = initvalues,
#                    eqcons=(), f_eqcons=None, ieqcons=(), f_ieqcons=None,
#                    bounds=RegionBounds, 
#                    fprime=None, fprime_eqcons=None, fprime_ieqcons=None, 
#                    args=(X), 
#                    iter=100, acc=1e-06, iprint=1, disp=None, full_output=0, epsilon=1.4901161193847656e-08, callback=None)
#    jac=jac,
#    OptL3L4 = sp.optimize.least_squares(RegressionEstimateL3L4_MaxRsq, x0=initvalues, 
#          bounds=RegionBounds, args=(X), verbose=1)
    OptL3L4 = sp.optimize.least_squares(RegressionEstimateL3L4_MaxRsq, x0=initvalues, 
          args=(X), verbose=1)
#    print(OptL3L4)
    L3              =   OptL3L4.x[0]
    L4              =   OptL3L4.x[1]
    [L1,L2]         =   RegressionEstimateL1L2(X,L3,L4)
    OptParameters   =   [L1,L2,L3,L4]
    return OptParameters

def setregionbounds(GLDregion):
    #set L4 and L4 bounds on GLD region
    tol = .00001
    if GLDregion == 1:
        bounds = [(None,-1.0-tol), (1.0+tol,None)]
    elif GLDregion == 2:
        bounds = [(1.0+tol,None), (None,-1.0-tol)]
    if GLDregion == 3:
        bounds = ((0,100),(0,100))
    elif GLDregion == 4:
        bounds = [(None,0.0-tol), (None,0.0-tol)]
    if GLDregion == 5:
        bounds = [(-1.0+tol, 0.0-tol), (1.0+tol,None)]
    elif GLDregion == 6:
        bounds = [(1.0+tol,None), (-1.0+tol,0.0-tol)]
        
    return bounds

def setinitvaluesL1L2L3L4():
    #set L4 and L4 bounds on GLD region
    tol         = .05
    initvalues  = np.nan*np.ones(3)
    initvalues[0] = .05
    initvalues[1] = .05
    initvalues[2] = .05         
    return initvalues

def setinitvalues(bounds):
    #set L4 and L4 bounds on GLD region
    tol         = .05
    initvalues  = np.nan*np.ones(2)
    for i in range(0,2):
        cbounds    = bounds[i]
        if cbounds[0] == None:
            cinit = cbounds[1]-tol #init L3 less than upper bound
        if cbounds[1] == None:
            cinit = cbounds[0]+tol #init L3 greater than lower bound 
        if cbounds[0]!=None and cbounds[1]!=None:
            cinit = (cbounds[0] + tol) #put bound in the middle
        initvalues[i] = cinit
    return initvalues
            
def  RegressionEstimateL1L2(X,L3,L4):
    X       =   np.sort(X)          #Sort X in accending order
    n       =   len(X)              #Number of observations in X
    n1      =   1.0*n
    R       =   np.nan*np.ones(n)    #Approximation of Q as per equation 16 page 82
    for i in range(0,n):
        R1      =   (i/(n1+1))**L3
        a       =  (n1-i+1)/(n1+1)
        R2      =   a**L4 
        R[i]    =   R1 - R2
    RR          =   [np.ones(n),R] #this is transposed the wrong way, see b = below
    #Look at this horror, just to do inv(x'*x)*x'y
    b           =   np.dot(np.dot(np.linalg.inv(np.dot(RR,np.transpose(RR))),RR),X)
    L1          =   b[0]
    L2          =   1.0/b[1]
    return L1,L2

def RegressionEstimateL3L4_MaxRsq(Parameters,*args):
    L3,L4       =   Parameters 
    X           =   args
    X       =   np.sort(X)          #Sort X in accending order
    n       =   len(X)              #Number of observations in X
    n1      = 1.0*n
    R       =   np.nan*np.ones(n)   #Approximation of Q as per equation 16 page 82
    for i in range(0,n):
        R1      =   (i/(n1+1))**L3
        R2      =   ((n1-i+1)/(n1+1))**L4
        R[i]    =   R1 - R2
#    [R,X]
    C       =   np.corrcoef(R,X)
    C12     = C[0,1]
    Objective = (1-abs(C12))*100000
    return Objective

def xxx(Parameters,X):
    L3,L4       =   Parameters 
#    SaveFile    = 'Xsaved.pkl'
#    file        = open(SaveFile, 'rb')
#    X           = pk.load(file)
#    file.close()
#    print(X)
    X       =   np.sort(X)          #Sort X in accending order
    n       =   len(X)              #Number of observations in X
    n1      =  1.0*n
    R       =   np.nan*np.ones(n)   #Approximation of Q as per equation 16 page 82
    R1      =   np.nan*np.ones(n)   #Approximation of Q as per equation 16 page 82
    R2      =   np.nan*np.ones(n)   #Approximation of Q as per equation 16 page 82
    for i in range(0,n):
        i1          = i*1.0
        R1[i]       =   (i1/(n1+1))**L3
        R2[i]       =   ((n1-i1+1)/(n1+1))**L4
        R[i]    =   R1[i] - R2[i]
#    [R,X]
    C       =   np.corrcoef(R,X)
    C12     = C[0,1]
#    Rsq     =   (C12**2)               #Sign changed because min is optimized
#    print([L3,L4,Rsq])
    return R1,R2,R,n,-C12

def DrawGeneralizedLambda(GLDparameters,Cases):
#Fitting Statistical Distributions by Zaven A. Karian and Edward J. Dudewicz
    R   =   {} #return R on R
    L1  =   GLDparameters[0]
    L2  =   GLDparameters[1]
    L3  =   GLDparameters[2]
    L4  =   GLDparameters[3]
    
    y       =   np.random.uniform(0,1,Cases)   #Uniform draws on (0,1)
    x       =   np.nan*np.ones(Cases)          #Allocate space for GLD draws with GLDparameters
    LLtrue  =   np.nan*np.ones(Cases)          #Log likelihood given the true GLD
    pdf     =   np.nan*np.ones(Cases)          #Density given the true GLD
    #Equation 1.2.1 page 9
    for i in range(0,Cases):
        x[i]        =   L1 + ((y[i]**L3 - (1-y[i])**L4)/L2)
        pdf[i]      =   L2/((L3*y[i]**(L3-1)))  + (L4*(1-y[i])**(L4-1))
        LLtrue[i]   =   np.log(pdf[i])
     
    A           =   (1/(1+L3)) - (1/(1+L4))                                   #Equation 2.1.16 page 45
    B           =   (1/(1+2*L3)) + (1/(1+2*L4)) - 2*beta(1+L3,1+L4)            #Equation 2.1.17 page 45
    C           =   (1/(1+3*L3)) - (1/(1+3*L4)) - 3*beta(1+2*L3,1+L4) + 3*beta(1+L3,1+2*L4)  #Equation 2.1.18 page 45
    
    D           =   (1/(1+4*L3)) + (1/(1+4*L4)) - 4*beta(1+3*L3,1+L4) + 6*beta(1+2*L3,1+2*L4) - 4*beta(1+L3,1+3*L4)  #Equation 2.1.19 page 45
                 
    a1          =   L1 + A/L2                                       #Mean of true GLD, equation 2.1.12 page 45 
    a2          =   (B - A*A)/(L2*L2)                               #Variance of true GLD, equation 2.1.13 page 45 
    s           =   np.sqrt(a2)
    a3          =   (C-3*A*B+2*(A*A*A))/((L2*L2*L2)*(s*s*s))        #Skewness of true GLD, equation 2.1.14 page 45 
    a4          =   (D-4*A*C +6*A*A*B -3*A*A*A*A)/((L2**4)*(s**4))  #Kurtosis of true GLD, equation 2.1.15 page 45 

    R['x']                =   x
    R['y']                 =   y
    R['pdf']               =   pdf
    R['trueLL']            =   LLtrue
    R['LLsum']              =   np.sum(LLtrue)
    R['A']                 =   A
    R['B']                 =   B
    R['C']                 =   C
    R['D']                 =   D
    
    R['L1']                =   L1
    R['L2']                =   L2
    R['L3']                =   L3
    R['L4']                =   L4
    R['TrueMean']          =   a1
    R['TrueVariance']      =   a2
    R['TrueSkewness']      =   a3
    R['TrueKurtosis']      =   a4
    R['SampleMean']        =   np.mean(x)
    R['SampleVariance']    =   np.var(x,ddof=1)
    R['SampleSkewness']    =   stats.skew(x,bias=False)
    R['SampleKurtosis']    =   stats.kurtosis(x,fisher=False,bias=False)
    return R

def LikelihoodGLD(GLDparameters,*args):
#Fitting Statistical Distributions by Zaven A. Karian and Edward J. Dudewicz
#    L1  =   GLDparameters[0]
    L2  =   GLDparameters[0]
    L3  =   GLDparameters[1]
    L4  =   GLDparameters[2]
    X           =   args
#    x       =   args   
    y           =   ut.ecdf(X)
    leny        =   len(y)
    y[leny-1]   =   y[leny-2]+(y[leny-1]-y[leny-2])/2.0
    Cases       =   len(X)
    LLtrue      =   np.nan*np.ones(Cases)          #Log likelihood given the true GLD
    pdf         =   np.nan*np.ones(Cases)          #Density given the true GLD
    #Equation 1.2.1 page 9
    for i in range(0,Cases):
        pdf[i]      =   L2/((L3*y[i]**(L3-1)))  + (L4*(1-y[i])**(L4-1))
        print([i,y[i],pdf[i]])
        if y[i] < 1.0:
            LLtrue[i]   =   np.log(pdf[i])
            
    return -abs(np.nansum(LLtrue))

def LikelihoodGLD2(GLDparameters,X):
#Fitting Statistical Distributions by Zaven A. Karian and Edward J. Dudewicz
#    L1  =   GLDparameters[0]
    R   = {}
    L2  =   GLDparameters[1]
    L3  =   GLDparameters[2]
    L4  =   GLDparameters[3]  
    y           =   ut.ecdf(X)
    leny        =   len(y)
    y[leny-1]   =   y[leny-2]+(y[leny-1]-y[leny-2])/2.0
    Cases       =   len(X)
    LLtrue      =   np.nan*np.ones(Cases)          #Log likelihood given the true GLD
    pdf         =   np.nan*np.ones(Cases)          #Density given the true GLD
    #Equation 1.2.1 page 9
    for i in range(0,Cases):
        pdf[i]      =   L2/((L3*y[i]**(L3-1)))  + (L4*(1-y[i])**(L4-1))
#        print([i,y[i],pdf[i]])
        if y[i] < 1.0:
            LLtrue[i]   =   np.log(pdf[i])
            

    R['X']          =   X
    R['y']          =   y
    R['pdf']        =   pdf
    R['trueLL']     =   LLtrue
    R['LLsum']      =   np.nansum(LLtrue)
    return R