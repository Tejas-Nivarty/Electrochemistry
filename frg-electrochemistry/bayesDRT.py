import numpy as np
import time
import matplotlib.pyplot as plt
from ReadDataFiles import readBayesDRT2, colorFader, readPEIS
import pandas as pd

from bayes_drt2.inversion import Inverter

def plotCompareNyquist(filenames,title,freqRange,legendList=None):
    
    numberOfPlots = len(filenames)
    circuitList = []
    fig, ax = plt.subplots()
    
    for i in range(0,numberOfPlots):
        
        #gets f and Z values
        f, Z = readBayesDRT2(filenames[i],freqRange)
        
        #plots results
        color = colorFader('blue','red',i,numberOfPlots)
        if legendList != None:
            ax.plot(Z.real,-Z.imag,'o',color=color,label=legendList[i])
        else:
            ax.plot(Z.real,-Z.imag,'o',color=color)
        
    
    maxBounds = max([ax.get_ylim()[1],ax.get_xlim()[1]])
    
    ax.set(title = title + ' Nyquist Plots',
           xlabel = 'Re(Z(f)) ($\Omega$)',
           ylabel = '-Im(Z(f)) ($\Omega$)')
    
    plt.axis('square')
    
    if legendList != None:
        ax.legend()
    
    plt.show()
    
    return circuitList

def getDRT(filename,title=None,freqRange=None,legendList=None):
    
    freq, Z = readBayesDRT2(filename,freqRange)
    
    
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
        
    return

def plotCompareDRT(filenames,title,freqRange,legendList=None,saveData=False):
    
    plotCompareNyquist(filenames,title,freqRange,legendList=legendList)
    
    freqRange = np.nan_to_num(freqRange,neginf=0.1,posinf=7e6)
    logFreqRange = -np.log10(freqRange)
    confInt = 95
    
    numberOfPlots = len(filenames)
    fig, ax = plt.subplots()
    ax.set_title(title)
    timeConstantArray = np.logspace(logFreqRange[0],logFreqRange[1], 1000)
    finaldf = None
    
    if saveData:
        
        finaldf = pd.DataFrame()
    
    for i in range(0,numberOfPlots):
        
        color = colorFader('blue','red',i,numberOfPlots)
        
        freq, Z = readBayesDRT2(filenames[i],freqRange)
        inv_hmc = Inverter(basis_freq=freq)
        inv_hmc.fit(freq, Z, mode='sample',nonneg=True,outliers=True)
        
        #use predict distribution, can adjust percentile to get error bars
        gammaMean = inv_hmc.predict_distribution(tau=timeConstantArray)
        gammaLo = inv_hmc.predict_distribution(tau=timeConstantArray,percentile=50-(confInt/2))
        gammaHi = inv_hmc.predict_distribution(tau=timeConstantArray,percentile=50+(confInt/2))
        
        if legendList == None:
            ax.plot(timeConstantArray,gammaMean,color=color)
        else:
            ax.plot(timeConstantArray,gammaMean,color=color,label=legendList[i])
            
        ax.fill_between(timeConstantArray,gammaLo,gammaHi,color=color,alpha=0.2,label='_')
        
        if saveData:
            
            finaldf[legendList[i]+' TimeConstant(s)'] = timeConstantArray
            finaldf[legendList[i]+' gammaLo(Ohm)'] = gammaLo
            finaldf[legendList[i]+' gammaHi(Ohm)'] = gammaHi
            finaldf[legendList[i]+' gammaMean(Ohm)'] = gammaMean
        
        continue
    
    if legendList != None:
        ax.legend()
    ax.set(ylabel = r'$\gamma$ ($\Omega$)',
            xlabel = r'Time Constant (s)',
            xscale='log')
    plt.show()
    
    return finaldf