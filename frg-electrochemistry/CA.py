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
            color = colorFader('green','blue',i,len(tafelList))
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

def plotCA(filenames,pH,area,referencePotential,title,legendList=None):
    
    fig, ax = plt.subplots()
    thermodynamicPotential = 0 #V vs. RHE
    
    for i, filename in enumerate(filenames):
        
        data = readCA(filename,pH,area,referencePotential)
        color = colorFader('blue','red',i,len(filenames))
        if legendList == None:
            overpotential = (thermodynamicPotential-data['Ewe/mV'].mean())
            label = '{:3.0f} mV'.format(overpotential)
        else:
            label = legendList[i]
            
        ax.plot(data['time/s'],
                data['j/mA*cm-2'],
                color=color,
                label=label)
    
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

tafel1FilenameList = [r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-04\Tafel-Ultra-tin-SRO-BTO-SRO-NSTO-1M KOH-Graphite Counter- SCE Ref-continous_04_CA_C01.txt',
                      r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-04\Tafel-Ultra-tin-SRO-BTO-SRO-NSTO-1M KOH-Graphite Counter- SCE Ref-continous_06_CA_C01.txt',
                      r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-04\Tafel-Ultra-tin-SRO-BTO-SRO-NSTO-1M KOH-Graphite Counter- SCE Ref-continous_08_CA_C01.txt',
                      r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-04\Tafel-Ultra-tin-SRO-BTO-SRO-NSTO-1M KOH-Graphite Counter- SCE Ref-continous_10_CA_C01.txt',
                      r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-04\Tafel-Ultra-tin-SRO-BTO-SRO-NSTO-1M KOH-Graphite Counter- SCE Ref-continous_12_CA_C01.txt',
                      r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-04\Tafel-Ultra-tin-SRO-BTO-SRO-NSTO-1M KOH-Graphite Counter- SCE Ref-continous_14_CA_C01.txt',
                      ]
tafel2FilenameList = [r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-05\Poed-up-Ultra Thin-SRO-BTO-SRO-NSTO-Graphite Counter-SCE Ref-1M KOH_05_CA_C01.txt',
                      r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-05\Poed-up-Ultra Thin-SRO-BTO-SRO-NSTO-Graphite Counter-SCE Ref-1M KOH_07_CA_C01.txt',
                      r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-05\Poed-up-Ultra Thin-SRO-BTO-SRO-NSTO-Graphite Counter-SCE Ref-1M KOH_09_CA_C01.txt',
                      r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-05\Poed-up-Ultra Thin-SRO-BTO-SRO-NSTO-Graphite Counter-SCE Ref-1M KOH_11_CA_C01.txt',
                      r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-05\Poed-up-Ultra Thin-SRO-BTO-SRO-NSTO-Graphite Counter-SCE Ref-1M KOH_13_CA_C01.txt',
                      r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-05\Poed-up-Ultra Thin-SRO-BTO-SRO-NSTO-Graphite Counter-SCE Ref-1M KOH_15_CA_C01.txt',
                      ]

plotCA(tafel1FilenameList,
       14,
       0.18,
       0.197,
       'Tafel 1')
plotCA(tafel2FilenameList,
       14,
       0.18,
       0.197,
       'Poled Up')


tafel1 = getTafel(tafel1FilenameList,
                  14,
                  0.18,
                  0.197)
tafel2 = getTafel(tafel2FilenameList,
                  14,
                  0.18,
                  0.197)

plotTafel([tafel1,
           tafel2],
          ['Unpoled (?)',
           'Poled-Up'],
          'Ultra-Thin SRO-BTO-SRO-NSTO')