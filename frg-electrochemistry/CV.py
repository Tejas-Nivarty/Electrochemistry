import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from ReadDataFiles import colorFader

def plotOneCV(data: pd.DataFrame, title: str):
    """Plots every cycle of one CV scan.

    Args:
        data (pd.DataFrame): DataFrame from readCV.
        title (str): Title of plot.

    Returns:
        tuple(matplotlib.figure.Figure,list[matplotlib.axes._axes.Axes]): fig and ax for further customization if necessary
    """
    numCycles = int(data['cycle number'].max())
    fig, ax = plt.subplots()
    ax.axhline(0,color='k')
    ax.axvline(0,color='k')
    for i in range(1,numCycles+1):
        color = colorFader('green','blue',i-1,numCycles)
        dataSlice = data[data['cycle number'] == i]
        ax.plot(dataSlice['Ewe/V'],dataSlice['j/mA*cm-2'],color = color,label = 'Cycle '+str(i))
    ax.set(title = title,
           xlabel = r'$V_{RHE}$',
           ylabel = r'j $(\frac{mA}{cm^2_{geo}})$')
    ax.legend()
    plt.show()
    return (fig, ax)
    
def plotManyCVs(dataList: list[pd.DataFrame], title: str, legendList: list[str] = None, cycleList: list[int] = None, horizontalLine: bool = False, verticalLine: bool = False, xlim: list[float] = None, ylim: list[float] = None, currentdensity: bool = True):
    """Plots one cycle (default last) of many CVs.

    Args:
        dataList (list[pd.DataFrame]): List of CV DataFrames.
        title (str): Title of plot.
        legendList (list[str], optional): List of legend items. Defaults to None.
        cycleList (list[int], optional): List of specific cycles to plot. Defaults to None.
        horizontalLine (bool, optional): Plots line through current = 0. Defaults to False.
        verticalLine (bool, optional): Plots line through V_RHE = 0. Defaults to False.
        xlim (list[float], optional): limits of x-axis. Defaults to None.
        ylim (list[float], optional): limits of y-axis. Defaults to None.
        currentdensity (bool, optional): If false, uses raw current. Defaults to True.

    Returns:
        tuple(matplotlib.figure.Figure,list[matplotlib.axes._axes.Axes]): fig and ax for further customization if necessary
    """
    
    numFiles = len(dataList)
    
    fig, ax = plt.subplots()
    
    if horizontalLine:
        ax.axhline(0,color='k')
    if verticalLine:
        ax.axvline(0,color='k')
        
    for i in range(numFiles):
        
        data = dataList[i]
        color = colorFader('blue','red',i,numFiles)
        
        try:
            numCycles = int(data['cycle number'].max())
        except ValueError: #empty dataframe, no data
            continue
        
        if cycleList == None: #if no specific cycle specified
            if numCycles == 1:
                cycle = 1
            else:
                cycle = numCycles - 1 #last cycle often gets cut off
        else:
            cycle = cycleList[i]
        dataSlice = data[data['cycle number'] == cycle]
        if legendList == None:
            
            #finds scan rate and puts that on legend
            dataSlice['Scan Rate (mV/s)'] = data['Ewe/mV'].diff()/data['time/s'].diff()
            dataSlicePositive = dataSlice[dataSlice['Scan Rate (mV/s)'] > 0]
            scanRate = dataSlicePositive['Scan Rate (mV/s)'].mean()
            
            if currentdensity:
                ax.plot(dataSlice['Ewe/mV'],dataSlice['j/mA*cm-2'],color = color,label='{:3.0f}'.format(scanRate)+r' $\frac{mV}{s}$')
            else:
                ax.plot(dataSlice['Ewe/mV'],dataSlice['I/A'],color = color,label='{:3.0f}'.format(scanRate)+r' $\frac{mV}{s}$')
        else:
            if currentdensity:
                ax.plot(dataSlice['Ewe/mV'],dataSlice['j/mA*cm-2'],color = color,label = legendList[i])
            else:
                ax.plot(dataSlice['Ewe/mV'],dataSlice['I/A'],color = color,label = legendList[i])
    if currentdensity:
        ax.set(title = title,
            xlabel = r'$mV_{RHE}$',
            ylabel = r'j $(\frac{mA}{cm^2_{geo}})$')
    else:
        ax.set(title = title,
            xlabel = r'$mV_{RHE}$',
            ylabel = r'Current (A)')

    ax.legend()
    
    if ylim:
        ax.set(ylim=ylim)
    if xlim:
        ax.set(xlim=xlim)

    plt.show()
    return (fig, ax)

def plotECSA(dataList: list[pd.DataFrame], title: str, trasatti: bool = False):
    """Takes list of DataFrames from buildEDLCList and finds ECSA.

    Args:
        dataList (list[pd.Dataframe]): CV DataFrames
        title (str): title of graph + EDLC
        trasatti (bool): set to False, implements Trasatti's method, unfinished

    Returns:
        tuple(matplotlib.figure.Figure,list[matplotlib.axes._axes.Axes]): fig and ax for further customization if necessary
    """
    
    currentList = []
    scanRateList = []
    
    for i, data in enumerate(dataList):
        
        numCycles = int(data['cycle number'].max())
        
        #takes last cycle
        if numCycles == 1:
            cycle = 1
        else:
            cycle = numCycles - 1
            
        #dataSlice stores data for last cycle
        dataSlice = data[data['cycle number'] == cycle]
        
        #finds scan rate
        dataSlice['Scan Rate (V/s)'] = data['Ewe/V'].diff()/data['time/s'].diff()
        dataSlicePositive = dataSlice[dataSlice['Scan Rate (V/s)'] > 0]
        scanRateList.append(dataSlicePositive['Scan Rate (V/s)'].mean())
        
        # Calculate first derivative
        voltage_diff = np.diff(dataSlice['control/V'])

        # Find where first derivative changes sign (this is the turning point)
        turn_idx = np.where(np.diff(np.sign(voltage_diff)))[0]

        # If multiple turning points are found, select the most prominent one
        # (typically the one with the largest absolute change in derivative)
        turn_idx = turn_idx[0]

        # Adjust index to match original data (np.diff reduces length by 1)
        turn_idx += 1

        # Split into forward and reverse scans
        forwardSlice = dataSlice.iloc[:turn_idx+1]
        reverseSlice = dataSlice.iloc[turn_idx:]
        
        plt.plot(dataSlice['time/s'],dataSlice['control/V'],'k')
        plt.plot(dataSlice['time/s'].iloc[0:-1],voltage_diff,'k')
        plt.show()
        
        #finds maximum oxidative and maximum reductive current
        forwardSliceLatestTimeIndex = forwardSlice['time/s'].idxmax()
        reverseSliceLatestTimeIndex = reverseSlice['time/s'].idxmax()
        #takes 9 largest values and removes last value to avoid spikes
        steadyStateForwardCurrent = forwardSlice.loc[forwardSliceLatestTimeIndex-10:forwardSliceLatestTimeIndex-1]['I/A'].mean()
        steadyStateReverseCurrent = reverseSlice.loc[reverseSliceLatestTimeIndex-10:reverseSliceLatestTimeIndex-1]['I/A'].mean()
        currentList.append(abs(steadyStateForwardCurrent-steadyStateReverseCurrent)/2)

    if not trasatti:
        
        #plots ECSA CVs
        legendList = ['{:3.0f} '.format(j*1000) + r'$\frac{mV}{s}$' for j in scanRateList]
        plotManyCVs(dataList,title+' EDLC CVs',legendList=legendList,horizontalLine=True,currentdensity=False)
        
        #performs linear regression and plots
        result = linregress(scanRateList,currentList)
        
        fig, ax = plt.subplots()
        ax.plot(scanRateList,currentList,'ko')
        xValues = np.linspace(0,ax.get_xlim()[1],3)
        yValues = (xValues * result.slope) + result.intercept
        ax.plot(xValues,yValues)
        ax.set(title = title + ' EDLC Plot, Slope = {:.5f} $\mu$F'.format(result.slope*1e6),
            xlabel = 'Scan Rate (V/s)',
            ylabel = 'Current (A)',
            xlim = [0,ax.get_xlim()[1]],
            ylim = [0,ax.get_ylim()[1]])
        plt.show()
        
    else: #incomplete for now
        xList = []
        yList = []
        
        for i,scanRate in enumerate(scanRateList):
            xValue = scanRate**0.5
            xList.append(xValue)
            yList.append(currentList[i]/xValue)
            
        #plots ECSA CVs
        legendList = ['{:3.0f} '.format(i*1000)+r'$\frac{mV}{s}$' for i in scanRateList]
        plotManyCVs(dataList,legendList,title+' EDLC CVs',horizontalLine=True)
            
        #performs linear regression
        result = linregress(xList,yList)
        k1 = result.slope
        k2 = result.intercept
    
    return (fig, ax)