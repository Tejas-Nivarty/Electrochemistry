import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from math import log10
from ReadDataFiles import readCA, colorFader

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


downTafelpast10 = getTafel([r'2024-04-09-TN-01-029\6_Tafel_Down_05_CA_C02.txt',
                            r'2024-04-09-TN-01-029\6_Tafel_Down_06_CA_C02.txt',
                            r'2024-04-09-TN-01-029\6_Tafel_Down_07_CA_C02.txt'],
                           14,0.0929119616)

upTafelpast10 = getTafel([r'2024-04-09-TN-01-029\12_Tafel_Up_05_CA_C02.txt',
                          r'2024-04-09-TN-01-029\12_Tafel_Up_06_CA_C02.txt',
                          r'2024-04-09-TN-01-029\12_Tafel_Up_07_CA_C02.txt'],
                         14,0.0929119616)

downTafel20 = getTafel([r'2024-04-17-TN-01-032\7_Tafel_PoledDown_04_CA_C02.txt',
                        r'2024-04-17-TN-01-032\7_Tafel_PoledDown_05_CA_C02.txt',
                        r'2024-04-17-TN-01-032\7_Tafel_PoledDown_06_CA_C02.txt',
                        r'2024-04-17-TN-01-032\7_Tafel_PoledDown_07_CA_C02.txt'],
                       14,0.0632532579)

upTafel20 = getTafel([r'2024-04-17-TN-01-032\14_Tafel_PoledUp_04_CA_C02.txt',
                      r'2024-04-17-TN-01-032\14_Tafel_PoledUp_05_CA_C02.txt',
                      r'2024-04-17-TN-01-032\14_Tafel_PoledUp_06_CA_C02.txt',
                      r'2024-04-17-TN-01-032\14_Tafel_PoledUp_07_CA_C02.txt'],
                    14,0.0632532579)

downTafelrecent10 = getTafel([r'2024-04-22-TN-01-033\13_Tafel_Poled_Down_05_CA_C02.txt',
                              r'2024-04-22-TN-01-033\13_Tafel_Poled_Down_06_CA_C02.txt',
                              r'2024-04-22-TN-01-033\13_Tafel_Poled_Down_07_CA_C02.txt',
                              r'2024-04-22-TN-01-033\13_Tafel_Poled_Down_08_CA_C02.txt',],
                             14,0.1277508776)

upTafelrecent10 = getTafel([r'2024-04-22-TN-01-033\7_Tafel_Poled_Up_05_CA_C02.txt',
                            r'2024-04-22-TN-01-033\7_Tafel_Poled_Up_06_CA_C02.txt',
                            r'2024-04-22-TN-01-033\7_Tafel_Poled_Up_07_CA_C02.txt',
                            r'2024-04-22-TN-01-033\7_Tafel_Poled_Up_08_CA_C02.txt'],
                           14,0.1277508776)


plotTafel([downTafelpast10,upTafelpast10,downTafelrecent10,upTafelrecent10,downTafel20,upTafel20],
          ['10 nm Down','10 nm Up','10 nm Down (new)','10 nm Up (new)','20 nm Down','20 nm Up'],
          'SRO Underlayer Series Static Tafel Slopes',
          colors=['green','darkgreen','blue','darkblue','red','darkred'])
