# Factor tester
import numpy as np
from scipy import stats
import UtilityFunctions as ut
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

#        print('Linear regression using stats.linregress')
#        print('parameters: a=%.2f b=%.2f \nregression: a=%.2f b=%.2f, std error= %.3f, r-square= %.3f, t-stat= %.3f'
#              % (alphaest,betaest,alphaest,betaest,stderr,rsq,tstat))


def FindTile(NumTile, Signal_AP):
    # NumTiles is 5 for quintiles 10 for deciles
    # Signal is the Asset by Period array of values to be assigned tiles
    [NumAsset, NumPeriod] = Signal_AP.shape
#    Tile_AP   = np.zeros([NumAsset,NumPeriod], dtype=np.int8) #Initialise tiles to zero
    Tile_AP = np.zeros([NumAsset, NumPeriod])  # Initialise tiles to zero
    for p in range(0, NumPeriod):
        Signal_p = Signal_AP[:, p].copy()  # Copy data to a vector for easy use
        [NotNaNvalues, NotNaninds] = ut.removenan(Signal_p)
        # Find number of companys that have Factor data
        NumNotNaN = len(NotNaNvalues)
        # Number of companies in each tile
        CompPerTile = np.ceil(NumNotNaN/NumTile)
        Index = np.argsort(NotNaNvalues)  # Index to sorted values
        CurrentTile = np.int_(1)  # Initialize the tile counter to 1
        for a in range(0, NumNotNaN):
            # If the company should go in the next tile
            if ut.rem(a+1, CompPerTile) == 0 and a != NumNotNaN-1:
                if CurrentTile != NumTile:  # Could be a few extra in the last tile, so stop the count
                    CurrentTile = CurrentTile + np.int_(1)
            # Assign tile to company for the month
            Tile_AP[NotNaninds[Index[a]], p] = CurrentTile
    return Tile_AP


def ReturnByTile(Tile_AP, Returns_AP):
    # Computes equally weighted returns by tile for a factor
    NumAsset, NumPeriod = Returns_AP.shape
    NumTile = np.int_(np.nanmax(Tile_AP))
    # Initialise returns by tile to zero
    RetByTile_pt = np.zeros([NumPeriod, NumTile])
    for p in range(0, NumPeriod):
        # Zero out the counter of companies by tile
        NumCompPerTile = np.zeros(NumTile)
        for a in range(0, NumAsset):
            if not np.isnan(Returns_AP[a, p]) and Tile_AP[a, p] != 0:
                TAP = np.int_(Tile_AP[a, p])  # Must be an integer
                RetByTile_pt[p, TAP-1] = RetByTile_pt[p, TAP-1] + \
                    Returns_AP[a, p]  # Assign returns to proper tile
                NumCompPerTile[TAP-1] = NumCompPerTile[TAP-1] + 1.0
        for t in range(0, NumTile):
            if NumCompPerTile[t] != 0:
                RetByTile_pt[p, t] = RetByTile_pt[p, t] / \
                    NumCompPerTile[t]  # Equal weight tile returns
    for p in range(0, NumPeriod):
        for t in range(0, NumTile):
            if RetByTile_pt[p, t] == 0:
                RetByTile_pt[p, t] = np.nan
    return RetByTile_pt


def CalcICByPeriod(Signal_AP, Returns_AP):
    MinNumForCorrelation = 5
    NumPeriod = Returns_AP.shape[1]
    ICByPeriod_p = np.nan*np.zeros(NumPeriod)
    ICByPeriodPval_p = np.nan*np.zeros(NumPeriod)
    # CumExcessRetBySignalTile        =   nan(NumTiles,NumPeriods);
    for p in range(0, NumPeriod):
        if p+1 <= NumPeriod-1:
            s = Signal_AP[:, p]
            f = Returns_AP[:, p+1]
            s, f, goodlist = ut.pairwise(s, f)
            if not ut.isempty(goodlist) and len(goodlist) >= MinNumForCorrelation:
                rho, pval = stats.spearmanr(s, f)
                ICByPeriod_p[p] = rho
                ICByPeriodPval_p[p] = pval
    GrandMeanIC = np.nanmean(ICByPeriod_p)
    GrandStdIC = np.nanstd(ICByPeriod_p)
    ICitp = GrandMeanIC/GrandStdIC
    return ICByPeriod_p, ICByPeriodPval_p, GrandMeanIC, GrandStdIC, ICitp


def CalcICdecay(Factor_AP, Returns_AP, NumForwardReturns):
    NumPeriod = Returns_AP.shape[1]
    FactorIC_pn = np.nan*np.zeros([NumPeriod, NumForwardReturns])
    FactorICPval_pn = np.nan*np.zeros([NumPeriod, NumForwardReturns])
    for p in range(0, NumPeriod):
        Factor_a = Factor_AP[:, p]
        for n in range(0, NumForwardReturns):
            fp = p + 1 + n  # forward period
            if fp <= NumPeriod - 1:  # Stay within the array
                forwardreturns_a = Returns_AP[:, fp]
                factor, forwardreturns_a, goodlist = ut.pairwise(
                    Factor_a, forwardreturns_a)
                if not ut.isempty(goodlist):
                    rho, pval = stats.spearmanr(factor, forwardreturns_a)
                    FactorIC_pn[p, n] = rho
                    FactorICPval_pn[p, n] = pval
    factor = np.array(range(NumForwardReturns, 0, -1))*1.0
    # Rule of 78 if  NumForwardReturns is 12
    ICdecayWeights_n = factor/sum(factor)
    ICdecay_n = np.nanmean(FactorIC_pn, axis=0)
    ICdecaySTD_n = np.nanstd(FactorIC_pn, axis=0)
    IRdecay_n = ICdecay_n/ICdecaySTD_n
    ICdecayWeighted = 0.0
    for i in range(0, len(ICdecayWeights_n)):
        ICdecayWeighted = ICdecayWeighted + ICdecayWeights_n[i]*ICdecay_n[i]
#    print('In CalcICdecay')
#    print(type(ICdecayWeighted))
    return ICdecay_n, ICdecaySTD_n, FactorIC_pn, FactorICPval_pn, IRdecay_n, ICdecayWeighted


def CalcTileStatistics(R):
    NumTile = R['NumTile']
    NumPeriod = R['NumPeriod']
    R['AlphaBySignalTile_t'] = np.nan*np.zeros(NumTile)
    R['BetaBySignalTile_t'] = np.nan*np.zeros(NumTile)
    R['RsquareBySignalTile_t'] = np.nan*np.zeros(NumTile)
    R['TstatBySignalTile_t'] = np.nan*np.zeros(NumTile)
    R['ResidualRiskBySignalTile_t'] = np.nan*np.zeros(NumTile)
    R['ResidualBySignalTile_pt'] = np.nan*np.zeros([NumPeriod, NumTile])
    for tile in range(0, NumTile):
        y = R['RetBySignalTile_pt'][:, tile]
        X = R['EW_BenchMark_p']
        y, X, goodlist = ut.pairwise(y, X)
        if not ut.isempty(goodlist) and len(goodlist) >= 5:
            alphaest, betaest, rsq, tstat, stderr = stats.linregress(
                y, X)  # Linear regression using stats.linregress
            R['AlphaBySignalTile_t'][tile] = alphaest
            R['BetaBySignalTile_t'][tile] = betaest
            R['RsquareBySignalTile_t'][tile] = rsq
            R['TstatBySignalTile_t'][tile] = tstat
            R['ResidualRiskBySignalTile_t'][tile] = stderr
            for p in range(0, len(goodlist)):
                R['ResidualBySignalTile_pt'][goodlist[p],
                                             tile] = y[p]-(alphaest+betaest*X[p])

    R['ExcessRetBySignalTile_pt'] = np.nan*np.zeros([NumPeriod, NumTile])
    R['CumulativeTileReturn_t'] = np.nan*np.zeros(NumTile)
    R['CumulativeTileExcessReturn_t'] = np.nan*np.zeros(NumTile)
    for p in range(0, NumPeriod):
        R['ExcessRetBySignalTile_pt'][p,
                                      :] = R['RetBySignalTile_pt'][p, :] - R['EW_BenchMark_p'][p]
    for tile in range(0, NumTile):
        rbst = R['ExcessRetBySignalTile_pt'][:, tile]
        erbst = R['ExcessRetBySignalTile_pt'][:, tile]
        rbst, goodlist = ut.removenan(rbst)
        if not ut.isempty(goodlist):
            cp = np.cumprod(1.0+rbst)
            R['CumulativeTileReturn_t'][tile] = cp[len(rbst)-1]
        erbst, goodlist = ut.removenan(erbst)
        if not ut.isempty(goodlist):
            cp = np.cumprod(1.0+erbst)
            R['CumulativeTileExcessReturn_t'][tile] = cp[len(erbst)-1]
    R['ExPostIR_t'] = (R['AlphaBySignalTile_t']/R['ResidualRiskBySignalTile_t']) * \
        np.sqrt(R['AnnPeriods'])  # ExPost Annualized Information Ratio
    return R


def AnalyzeFactor(SignalName, Signal_AP, Returns_AP, NumTile):
    R = {}
    R['SignalName'] = SignalName
    R['Signal_AP'] = Signal_AP.copy()
    R['Returns_AP'] = Returns_AP.copy()
    R['NumAsset'] = Returns_AP.shape[0]
    R['NumPeriod'] = Returns_AP.shape[1]
    R['AnnPeriods'] = 12
    R['MinNumForCorrelation'] = 10
    R['NumForwardReturns'] = 12  # Number of forward returns for IC decay
    R['NumTile'] = np.int_(NumTile)
    # equaly weighted mean by period
    R['EW_BenchMark_p'] = np.nanmean(Returns_AP, axis=0)
    R['SignalTile_AP'] = FindTile(NumTile, R['Signal_AP'])
    R['ReturnTile_AP'] = FindTile(NumTile, R['Returns_AP'])
    R['RetBySignalTile_pt'] = ReturnByTile(R['SignalTile_AP'], R['Returns_AP'])
    R['ICByPeriod_p'], R['ICByPeriodPval_p'], R['GrandMeanIC'], R['GrandStdIC'], R['ICitp'] = CalcICByPeriod(
        R['Signal_AP'], R['Returns_AP'])
    R['ICdecay_n'], R['ICdecaySTD_n'], R['FactorIC_pn'], R['FactorICPval_pn'], R['IRdecay_n'], R['ICdecayWeighted'] = CalcICdecay(
        R['Signal_AP'], R['Returns_AP'], R['NumForwardReturns'])
    R = CalcTileStatistics(R)
    return R


def Optimize_IRitp_2(F1_AP, F2_AP, Returns_AP, NumPoints):
    S = {}
    NumForwardReturns = 12
    ICitp = np.nan*np.zeros(NumPoints)
    ICdecayWeighted = np.nan*np.zeros(NumPoints)
    ICdecay_p = np.nan*np.zeros([NumPoints, NumForwardReturns])
    Weights = np.nan*np.zeros([NumPoints, 2])

    for gridpoint in range(0, NumPoints):
        w1 = (1.0*gridpoint)/(1.0*NumPoints)
        w2 = 1.0-w1
        print('Processing point ' + str(gridpoint+1) +
              ' of ' + str(NumPoints) + ' w1 = ' + str(w1))
        Signal_AP = w1*F1_AP + w2*F2_AP
        ICByPeriod_p, ICByPeriodPval_p, GrandMeanIC, GrandStdIC, ICitp1 = CalcICByPeriod(
            Signal_AP, Returns_AP)
        ICdecay_n, ICdecaySTD_n, FactorIC_pn, FactorICPval_pn, IRdecay_n, ICdecayWeighted1 = CalcICdecay(
            Signal_AP, Returns_AP, NumForwardReturns)
#        print((gridpoint,ICitp1,GrandMeanIC,GrandStdIC))
        ICitp[gridpoint] = ICitp1
        ICdecay_p[gridpoint, :] = ICdecay_n
        ICdecayWeighted[gridpoint] = ICdecayWeighted1
        Weights[gridpoint, 0] = w1
        Weights[gridpoint, 1] = w2
    maxICitp = np.max(ICitp)
    # find not nan positions
    maxICitpindex = ut.indices(ICitp, lambda x: x == maxICitp)
    OptimalWeightsICitp = Weights[maxICitpindex, :][0]
    S['OptimalFactorICitp'] = OptimalWeightsICitp[0] * \
        F1_AP + OptimalWeightsICitp[1]*F2_AP
    S['OptimalWeightsICitp'] = OptimalWeightsICitp
    S['OptimalICitp'] = maxICitp
    S['ICdecay_p'] = ICdecay_p

    maxICdecayWeighted = np.max(ICdecayWeighted)
    maxICdecayWeightedindex = ut.indices(
        ICdecayWeighted, lambda x: x == maxICdecayWeighted)  # find not nan positions
    OptimalWeightsICdecayWeighted = Weights[maxICdecayWeightedindex, :][0]
    S['OptimalFactorICdecayWeighted'] = OptimalWeightsICdecayWeighted[0] * \
        F1_AP + OptimalWeightsICdecayWeighted[1]*F2_AP
    S['OptimalWeightsICdecayWeighted'] = OptimalWeightsICdecayWeighted
    S['OptimalICdecayWeighted'] = maxICdecayWeighted
    S['Weights'] = Weights
    S['ICitp'] = ICitp
    S['ICdecayWeighted'] = ICdecayWeighted
    S['OptimalICdecay'] = ICdecay_p[maxICdecayWeightedindex, :]
    return S


def Optimize_IRitp_3(F1_AP, F2_AP, F3_AP, Returns_AP, NumPoints):
    S = {}
    NumForwardReturns = 12
    ICitp = np.nan*np.zeros(NumPoints)
    ICdecayWeighted = np.nan*np.zeros(NumPoints)
    ICdecay_p = np.nan*np.zeros([NumPoints, NumForwardReturns])
    Weights = np.nan*np.zeros([NumPoints, 3])
    for gridpoint in range(0, NumPoints):
        w = np.random.uniform(0, 1, 3)
        w = w/sum(w)
        print('Processing point ' + np.str(gridpoint+1) + ' of ' + np.str(NumPoints) +
              '  w1 = ' + np.str(w[0]) + '  w2 = ' + np.str(w[1]) + '  w3 = ' + np.str(w[2]))
        Signal_AP = w[0]*F1_AP + w[1]*F2_AP + w[2]*F3_AP
        ICByPeriod_p, ICByPeriodPval_p, GrandMeanIC, GrandStdIC, ICitp1 = CalcICByPeriod(
            Signal_AP, Returns_AP)
        ICdecay_n, ICdecaySTD_n, FactorIC_pn, FactorICPval_pn, IRdecay_n, ICdecayWeighted1 = CalcICdecay(
            Signal_AP, Returns_AP, NumForwardReturns)
        ICitp[gridpoint] = ICitp1
        ICdecay_p[gridpoint, :] = ICdecay_n
        ICdecayWeighted[gridpoint] = ICdecayWeighted1
        Weights[gridpoint, 0] = w[0]
        Weights[gridpoint, 1] = w[1]
        Weights[gridpoint, 2] = w[2]
    maxICitp = np.max(ICitp)
    # find not nan positions
    maxICitpindex = ut.indices(ICitp, lambda x: x == maxICitp)
    print(maxICitpindex)
    OptimalWeightsICitp = Weights[maxICitpindex, :]
    # because the above line returns a list within a list, UHG!
    OptimalWeightsICitp = OptimalWeightsICitp[0]
#    print(OptimalWeightsICitp)
    w1 = OptimalWeightsICitp[0]
    w2 = OptimalWeightsICitp[1]
    w3 = OptimalWeightsICitp[2]
    S['OptimalFactorICitp'] = w1*F1_AP + w2*F2_AP + w3*F3_AP
    S['OptimalWeightsICitp'] = OptimalWeightsICitp
    S['OptimalICitp'] = maxICitp
    maxICdecayWeighted = np.max(ICdecayWeighted)
    maxICdecayWeightedindex = ut.indices(
        ICdecayWeighted, lambda x: x == maxICdecayWeighted)  # find not nan positions
    print(maxICdecayWeightedindex)
    OptimalWeightsICdecayWeighted = Weights[maxICdecayWeightedindex, :]
    OptimalWeightsICdecayWeighted = OptimalWeightsICdecayWeighted[0]
#    print(OptimalWeightsICdecayWeighted)
    w1 = OptimalWeightsICdecayWeighted[0]
    w2 = OptimalWeightsICdecayWeighted[1]
    w3 = OptimalWeightsICdecayWeighted[2]
    S['OptimalFactorICdecayWeighted'] = w1*F1_AP + w2*F2_AP + w3*F3_AP
    S['OptimalWeightsICdecayWeighted'] = OptimalWeightsICdecayWeighted
    S['OptimalICdecayWeighted'] = maxICdecayWeighted
    S['Weights'] = Weights
    S['ICitp'] = ICitp
    S['ICdecayWeighted'] = ICdecayWeighted
    S['OptimalICdecay'] = ICdecay_p[maxICdecayWeightedindex, :]
    return S


def Optimize_IRitp_n(FactorList, Returns_AP):
    NumberWeights = len(FactorList)
    X0          = np.ones(NumberWeights)/NumberWeights  # Initial weightslist
    X0[0] = .3
    X0[1] = .7
    bndslist    = []
    for i in range(0,NumberWeights):
        bndslist.append((0,None))
    bnds        = tuple(bndslist)
    A           = np.array(np.ones((1,NumberWeights)))
    cons        = LinearConstraint(A, lb=0, ub=1, keep_feasible=True)
    tol1 = .0000001
    res         = minimize(GetICitp, X0, args=(FactorList,Returns_AP),method='SLSQP',
                   bounds=bnds,constraints=cons,tol=tol1)
    return res

def GetICitp(x,FactorList,Returns_AP):
    Signal_AP = FactorList[0]*0
    for i in range(0,len(FactorList)):
        Signal_AP = Signal_AP + FactorList[i]*x[i]
    ICByPeriod_p, ICByPeriodPval_p, GrandMeanIC, GrandStdIC, ICitp = CalcICByPeriod(
    Signal_AP, Returns_AP)
    print(len(FactorList))
    print(x)
    print(ICitp*10000)
    return -ICitp*10000