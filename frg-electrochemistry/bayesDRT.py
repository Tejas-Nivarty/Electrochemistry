import numpy as np
import time
import matplotlib.pyplot as plt
from ReadDataFiles import colorFader, convertToImpedanceAnalysis
#from PEIS import plotManyNyquists
import pandas as pd

from bayes_drt2.inversion import Inverter

def getDRT(eisData: pd.DataFrame):
    """Gets single DRT of both HMC and MAP methods. From bayesDRT2 default functions.

    Args:
        eisData (pd.DataFrame): DataFrame of EIS data.
        
    Returns:
        tuple(matplotlib.figure.Figure,matplotlib.axes._axes.Axes): fig and ax for further customization if necessary
    """
    freq, Z = convertToImpedanceAnalysis(eisData)
    
    "Fit the data"
    # By default, the Inverter class is configured to fit the DRT (rather than the DDT)
    # Create separate Inverter instances for HMC and MAP fits
    # Set the basis frequencies equal to the measurement frequencies 
    # (not necessary in general, but yields faster results here - see Tutorial 1 for more info on basis_freq)
    inv_hmc = Inverter(basis_freq=freq)
    inv_map = Inverter(basis_freq=freq)

    # Perform HMC fit
    start = time.time()
    inv_hmc.fit(freq, Z, mode='sample',nonneg=True,outliers='auto')
    elapsed = time.time() - start
    print('HMC fit time {:.1f} s'.format(elapsed))

    # Perform MAP fit
    start = time.time()
    inv_map.fit(freq, Z, mode='optimize',nonneg=True,outliers='auto')  # initialize from ridge solution
    elapsed = time.time() - start
    print('MAP fit time {:.1f} s'.format(elapsed))
    
    "Visualize DRT and impedance fit"
    # plot impedance fit and recovered DRT
    fig,axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # plot fits of impedance data
    inv_hmc.plot_fit(axes=axes[0], plot_type='nyquist', color='k', label='HMC fit', data_label='Data')
    inv_map.plot_fit(axes=axes[0], plot_type='nyquist', color='r', label='MAP fit', plot_data=False)

    # add Dirac delta function for RC element
    axes[1].plot([np.exp(-2),np.exp(-2)],[0,10],ls='--',lw=1)

    # Plot recovered DRT at given tau values
    inv_hmc.plot_distribution(ax=axes[1], color='k', label='HMC mean', ci_label='HMC 95% CI')
    inv_map.plot_distribution(ax=axes[1], color='r', label='MAP')

    # axes[1].set_ylim(0,3.5)
    # axes[1].legend()


    fig.tight_layout()
    plt.show()
    
    # "Visualize the recovered error structure"
    # # For visual clarity, only MAP results are shown.
    # # HMC results can be obtained in the same way
    # fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharex=True)

    # # plot residuals and estimated error structure
    # inv_map.plot_residuals(axes=axes)

    # # plot true error structure in miliohms
    # p = axes[0].plot(freq, 3*Zdf['sigma_re'] * 1000, ls='--')
    # axes[0].plot(freq, -3*Zdf['sigma_re'] * 1000, ls='--', c=p[0].get_color())
    # axes[1].plot(freq, 3*Zdf['sigma_im'] * 1000, ls='--')
    # axes[1].plot(freq, -3*Zdf['sigma_im'] * 1000, ls='--', c=p[0].get_color(), label='True $\pm 3\sigma$')

    # axes[1].legend()

    # fig.tight_layout()
        
    return (fig, axes)

def plotManyDRTs(eisDatas: list[pd.DataFrame], title: str, legendList: list[str] = None, logResistance = False):
    """Plots many DRTs.

    Args:
        eisDatas (list[pd.DataFrame]): List of EIS DataFrames.
        title (str): Title of DRT plot.
        legendList (list[str], optional): List of legend elements. Defaults to potential that EIS was taken at.
        logResistance (bool, optional): Plots resistance in log format for better visualizing series.

    Returns:
        tuple(matplotlib.figure.Figure,matplotlib.axes._axes.Axes): fig and ax for further customization if necessary
    """
    #plotManyNyquists(eisDatas,title,freqRange,legendList=legendList)
    
    #automatically finds minimum and maximum of freqRange
    minFreq = np.inf
    maxFreq = -np.inf
    for eisData in eisDatas:
        currMinFreq = eisData['freq/Hz'].min()
        currMaxFreq = eisData['freq/Hz'].max()
        if currMinFreq < minFreq:
            minFreq = currMinFreq
        if currMaxFreq > maxFreq:
            maxFreq = currMaxFreq
    freqRange = [minFreq,maxFreq]
    
    freqRange = np.nan_to_num(freqRange,neginf=0.1,posinf=7e6)
    logFreqRange = -np.log10(freqRange)
    confInt = 95
    
    numberOfPlots = len(eisDatas)
    fig, ax = plt.subplots()
    ax.set_title(title)
    timeConstantArray = np.logspace(logFreqRange[0],logFreqRange[1], 1000)
    
    for i in range(0,numberOfPlots):
        
        color = colorFader('blue','red',i,numberOfPlots)
        
        freq, Z = convertToImpedanceAnalysis(eisDatas[i])
        inv_hmc = Inverter(basis_freq=freq)
        inv_hmc.fit(freq, Z, mode='sample',nonneg=True,outliers=True)
        
        #use predict distribution, can adjust percentile to get error bars
        gammaMean = inv_hmc.predict_distribution(tau=timeConstantArray)
        gammaLo = inv_hmc.predict_distribution(tau=timeConstantArray,percentile=50-(confInt/2))
        gammaHi = inv_hmc.predict_distribution(tau=timeConstantArray,percentile=50+(confInt/2))
        
        if legendList == None:
            #finds potential at which EIS was taken
            potential = eisDatas[i]['<Ewe>/V'].mean()*1000
            ax.plot(timeConstantArray,gammaMean,color=color,label='{:3.0f}'.format(potential)+r' $mV_{ref}$')
        else:
            ax.plot(timeConstantArray,gammaMean,color=color,label=legendList[i])
            
        ax.fill_between(timeConstantArray,gammaLo,gammaHi,color=color,alpha=0.2,label='_')
        
        continue
    
    ax.legend()
    ax.set(ylabel = r'$\gamma$ ($\Omega$)',
            xlabel = r'Time Constant (s)',
            xscale='log')
    
    if logResistance:
        ax.set_yscale('symlog', linthresh=1)
        ax.set_ylim([0,ax.get_ylim()[1]])
    
    plt.show()
    
    return (fig, ax)