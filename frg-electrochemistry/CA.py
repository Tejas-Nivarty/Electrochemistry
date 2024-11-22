import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from math import log10
from ReadDataFiles import readCA, colorFader, calculateIntegral

def getTafel(filenameList,pH,area,referencePotential):
    
    thermodynamicPotential = 0 #V vs. RHE
    
    overpotentialList = []
    logCurrentDensityList = []
    
    for file in filenameList:
        data = readCA(file,pH,area,referencePotential)
        #excludes first 100 seconds of data
        dataSlice = data[data['time/s'] > (data['time/s'].iloc[-1] - 500)]
        overpotentialList.append(thermodynamicPotential-dataSlice['Ewe/mV'].mean())
        logCurrentDensityList.append(log10(-dataSlice['j/mA*cm-2'].mean()))
        
    linearRegression = linregress(logCurrentDensityList,overpotentialList)
    
    
    
    return (overpotentialList,logCurrentDensityList,linearRegression)

def plotTafel(tafelList: list[tuple],legendList,title,colors=None):
    
    fig,ax = plt.subplots()
    maxLims = [0,0]
    
    for i in range(len(tafelList)):
        
        overpotentialList,logCurrentDensityList,linearRegression = tafelList[i]
        if colors == None:
            color = colorFader('green','blue',i/len(tafelList))
        else:
            color = colors[i]
        tafelSlope = linearRegression.slope
        exchangeCurrentDensity = 10**(-linearRegression.intercept/linearRegression.slope)
        tafelSlopeString = ', A = {:.3e} '.format(tafelSlope) + r'$\frac{mV}{dec}$'
        exchangeCurrentDensityString = r', $j_0$ =' + '{:.3e} '.format(exchangeCurrentDensity) + r'$\frac{mA}{cm^2_{geo}}$'
        
        ax.plot(logCurrentDensityList,
                overpotentialList,
                color = color,
                marker = 'o',
                label=legendList[i]+tafelSlopeString+exchangeCurrentDensityString)
        
        if (ax.get_ylim()[0] > maxLims[0]) or (ax.get_ylim()[1] > maxLims[1]):
            maxLims = ax.get_ylim()
        
        yValues = np.linspace(0,ax.get_ylim()[1]+20,3)
        xValues = (yValues - linearRegression.intercept)/linearRegression.slope
        ax.plot(xValues,yValues,color=color,linestyle='--',label='_')
        ax.set_ylim(maxLims)
        
    ax.set(title = title,
           xlabel = r'log(j $\frac{mA}{cm^2_{geo}}$)',
           ylabel = '$\eta$ (mV)',
           ylim = [0,ax.get_ylim()[1]])
    ax.legend()
    plt.show()
    
    return

def plotCA(filenames,pH,area,referencePotential,legendList,title):
    
    fig, ax = plt.subplots()
    
    for i, filename in enumerate(filenames):
        
        data = readCA(filename,pH,area,referencePotential)
        color = colorFader('blue','red',i,len(filenames))
        ax.plot(data['time/s'],
                data['j/mA*cm-2'],
                color=color,
                label=legendList[i])
    
    ax.set(title=title+' Chronoamperometry',
           ylabel = r'Current Density $(\frac{mA}{cm^2_{geo}})$',
           xlabel = 'Time (s)')
    ax.legend()
    
    plt.show()

    return

def integratePeaks(filenames):
    
    molesList = []
    
    for filename in filenames:
        data = readCA(filename,14,0.1,0.197)
        faradayConstant = 9.648533212331E4
        coulombsTransferred = calculateIntegral(data['time/s'],
                                                data['I/mA']/1000,
                                                0,
                                                [-np.inf,np.inf])
        molesTransferred = coulombsTransferred/faradayConstant
        print('{filename}: {moles} mol e-'.format(filename=filename,
                                                  moles=molesTransferred))
        
    
    return molesList

plotCA([r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-11-12-TN-01-058\11_Static_CA_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-11-12-TN-01-058\13_Static_CA_Booster_Stirring_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-11-13-TN-01-060\7_Static_CA_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-11-16-TN-01-061\7_Static_CA_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-11-20-TN-01-062\7_Static_CA_C01.txt'],
       14,
       0.1734377200,
       0.217,
       ['1','2','3','4','5'],
       r'TN-01-049-3, $V_{RHE}$ = -0.337')