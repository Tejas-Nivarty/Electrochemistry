import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from math import log10
from ReadDataFiles import readCA, colorFader, calculateIntegral
import pandas as pd

def getTafel(filenameList: list[str], pH: float, area: float, referencePotential: float):
    """Gets Tafel slope from list of CAs.

    Args:
        filenameList (list[str]): Filenames of CAs.
        pH (float): pH of electrolyte.
        area (float): Area of working electrode in cm^2.
        referencePotential (float): Reference electrode potential in V vs. SHE.

    Returns:
        tuple[list[float],list[float],LinRegressResult]: (overpotentialList,logCurrentDensityList,LinearRegression)
    """
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

def plotTafel(tafelList: list[tuple], legendList: list[str], title: str, colors: list[str] = None):
    """Plots list of Tafel slopes from one or many getTafel outputs.

    Args:
        tafelList (list[tuple]): List of tuples from getTafel.
        legendList (list[str]): Legend list to describe tuples.
        title (str): Title of plot overall.
        colors (list[str], optional): Custom colors to use for each tuple. Defaults to None.

    Returns:
        tuple(matplotlib.figure.Figure,list[matplotlib.axes._axes.Axes]): fig and ax for further customization if necessary
    """
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
    
    return (fig,ax)

def plotCA(caDatas: list[pd.DataFrame], title: str, legendList: list[str] = None):
    """Plots multiple CAs.

    Args:
        caDatas (list[pd.DataFrame]): List of CAs from readCAs.
        title (str): Title of plot.
        legendList (list[str], optional): Legend list. Defaults to None.
        
    Returns:
        tuple(matplotlib.figure.Figure,list[matplotlib.axes._axes.Axes]): fig and ax for further customization if necessary
    """
    fig, ax = plt.subplots()
    thermodynamicPotential = 0 #V vs. RHE
    
    for i, data in enumerate(caDatas):
           
        color = colorFader('blue','red',i,len(caDatas))
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

    return (fig, ax)

def integrateCA(caDatas: list[pd.DataFrame]):
    """Integrates CA data to find total charge transferred.

    Args:
        caDatas (list[pd.DataFrame]): List of CA DataFrames.

    Returns:
        list[tuple[float,float]] = (mol e-/cm^2, expTime (s)): Returns experiment time and moles transferred per area.
    """
    
    molesList = []
    
    for i, data in enumerate(caDatas):
        
        faradayConstant = 9.648533212331E4
        coulombsTransferredPerArea = calculateIntegral(data['time/s'],
                                                       data['j/mA*cm-2']/1000,
                                                       0,
                                                       [-np.inf,np.inf])
        molesTransferredPerArea = coulombsTransferredPerArea/faradayConstant
        experimentTime = data['time/s'].max()
        molesList.append((molesTransferredPerArea,experimentTime))
        
    return molesList

def plotH2CA(h2List,electronList,title,labels=None):
    """Plots integrated CA charge and H2 generated.

    Args:
        h2List (list[tuple[float,float]]): Dictionary from readExcelSheet. Must match order of electronList
        electronList (list[tuple[float,float]]): Dictionary from integrateCA. Must match order of h2Dict.
        title (str): Graph title.
        labels (list[str]): X-axis labels for experiments. Default None.
        
    Returns:
        tuple(matplotlib.figure.Figure,list[matplotlib.axes._axes.Axes]): fig and ax for further customization if necessary
    """
    
    h2prodList = []
    h2errList = []
    chargeList = []
    fig, ax = plt.subplots()
    
    for i, electronTuple in enumerate(electronList):
        
        h2prod, h2err = h2List[i]
        charge, time = electronTuple
        
        h2prod = h2prod*1E9/time #nmol/s/cm^2
        h2err = h2err*1E9/time #nmol/s/cm^2
        charge = abs(charge*1E9/time/2) #nmol/s/cm^2, charge divided by 2 for stoich
        
        h2prodList.append(h2prod)
        h2errList.append(h2err)
        chargeList.append(charge)
        
    xList = [i+1 for i in range(len(electronList))]
    
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
    
    return (fig, ax)