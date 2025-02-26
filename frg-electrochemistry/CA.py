import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from math import log10
from ReadDataFiles import readCA, colorFader, calculateIntegral
import re

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

def integrateCA(filenames):
    """Integrates CA data to find total charge transferred.

    Args:
        filenames (list[str]): List of CA filenames.

    Returns:
        dict[experimentNumber] = (mol e-, expTime (s)): Returns experiment time and moles transferred.
    """
    
    molesDict = {}
    
    for filename in filenames:
        
        data = readCA(filename,1,1,1)
        
        filename = filename.split('\\')[-1]
        
        experimentNumber = int(re.search(r'\d+', str(filename)).group())
        faradayConstant = 9.648533212331E4
        coulombsTransferred = calculateIntegral(data['time/s'],
                                                data['I/mA']/1000,
                                                0,
                                                [-np.inf,np.inf])
        molesTransferred = coulombsTransferred/faradayConstant
        experimentTime = data['time/s'].max()
        molesDict[experimentNumber] = (molesTransferred,experimentTime)
        
    
    return molesDict

def plotH2CA(h2Dict,electronDict,area,title,labels=None):
    """Plots integrated CA charge and H2 generated.

    Args:
        h2Dict (dict[key]): Dictionary from readExcelSheet. Must match keys in electronDict.
        electronDict (dict[key]): Dictionary from integrateCA. Must match keys in h2Dict.
        area (float): Area of electrode in cm^2.
        labels (list[str]): X-axis labels for experiments.
        title (str): Graph title.
    """
    
    h2prodList = []
    h2errList = []
    chargeList = []
    fig, ax = plt.subplots()
    
    for key in h2Dict:
        
        h2prod, h2err = h2Dict[key]
        charge, time = electronDict[key]
        
        h2prod = h2prod*1E9/time/area #nmol/s/cm^2
        h2err = h2err*1E9/time/area #nmol/s/cm^2
        charge = abs(charge*1E9/time/area/2) #nmol/s/cm^2, charge divided by 2 for stoich
        
        h2prodList.append(h2prod)
        h2errList.append(h2err)
        chargeList.append(charge)
        
    xList = [i+1 for i in range(len(h2Dict))]
    
    ax.errorbar(xList,
                h2prodList,
                yerr=h2errList,
                fmt='o',
                color='blue',
                capsize=5,
                capthick=1,
                label=r'$H_2$ Generated')
    ax.scatter(xList,
               chargeList, 
               color='goldenrod', 
               label=r'$\frac{1}{2}e^-$ Transferred')
    
    if labels != None:
        ax.set(title = title,
            ylabel = r'Mole Flux ($\frac{nmol}{cm^2\cdot s}$)',
            xticks = xList,
            xticklabels = labels,
            ylim = [0,ax.get_ylim()[1]])
    else:
        ax.set(title = title,
            ylabel = r'Mole Flux ($\frac{nmol}{cm^2\cdot s}$)',
            ylim = [0,ax.get_ylim()[1]])
        
    ax.legend()
    
    plt.show()
    
    return