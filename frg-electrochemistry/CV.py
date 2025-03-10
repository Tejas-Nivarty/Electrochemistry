import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
from ReadDataFiles import readCV, colorFader

def plotOneCV(data,title):
    numCycles = int(data['cycle number'].max())
    fig, ax = plt.subplots()
    ax.axhline(0,color='k')
    ax.axvline(0,color='k')
    for i in range(1,numCycles+1):
        color = colorFader('green','blue',0,1)
        dataSlice = data[data['cycle number'] == i]
        ax.plot(dataSlice['Ewe/V'],dataSlice['j/mA*cm-2'],color = color,label = 'Cycle '+str(i))
    ax.set(title = title,
           xlabel = r'$V_{RHE}$',
           ylabel = r'j $(\frac{mA}{cm^2_{geo}})$')
    ax.legend()
    plt.show()
    return
    
def plotCompareCV(dataList,legendList,title,cycleList=None,horizontalLine=False,verticalLine=False,xlim=None,ylim=None,currentdensity=True):
    numFiles = len(dataList)
    fig, ax = plt.subplots()
    if horizontalLine:
        ax.axhline(0,color='k')
    if verticalLine:
        ax.axvline(0,color='k')
    for i in range(numFiles):
        data = dataList[i]
        color = colorFader('blue','red',i,numFiles)
        numCycles = int(data['cycle number'].max())
        if cycleList == None:
            if numCycles == 1:
                cycle = 1
            else:
                cycle = numCycles - 1
        else:
            cycle = cycleList[i]
        dataSlice = data[data['cycle number'] == cycle]
        if legendList == None:
            if currentdensity:
                ax.plot(dataSlice['Ewe/mV'],dataSlice['j/mA*cm-2'],color = color)
            else:
                ax.plot(dataSlice['Ewe/mV'],dataSlice['I/A'],color = color)
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
    if legendList != None:
        ax.legend()
    if ylim:
        ax.set(ylim=ylim)
    if xlim:
        ax.set(xlim=xlim)

    plt.show()
    return

def plotECSA(dataList,title,trasatti=False):
    """Takes list of DataFrames from buildEDLCList and finds ECSA.

    Args:
        dataList (list[pd.Dataframe]): CV DataFrames
        title (str): title of graph + EDLC

    Returns:
        (tuple(LinregressResult,list[float],list[float])): (result,scanRateList,currentList)
    """
    numFiles = len(dataList)
    currentList = []
    scanRateList = []
    
    for i in range(numFiles):
        
        data = dataList[i]
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
        
        #uses control voltage to split into anodic and cathodic sweeps
        # Find the turning point (where voltage changes direction)
        voltage_diff = np.diff(dataSlice['control/V'])
        turn_idx = np.where(np.diff(np.signbit(voltage_diff)))[0][0]
        # Split into forward and reverse scans
        forwardSlice = dataSlice.iloc[:turn_idx+1]
        reverseSlice = dataSlice.iloc[turn_idx:]
        
        #finds maximum oxidative and maximum reductive current
        forwardSliceLatestTimeIndex = forwardSlice['time/s'].idxmax()
        reverseSliceLatestTimeIndex = reverseSlice['time/s'].idxmax()
        #takes 9 largest values and removes last value to avoid spikes
        steadyStateForwardCurrent = forwardSlice.loc[forwardSliceLatestTimeIndex-10:forwardSliceLatestTimeIndex-1]['I/A'].mean()
        steadyStateReverseCurrent = reverseSlice.loc[reverseSliceLatestTimeIndex-10:reverseSliceLatestTimeIndex-1]['I/A'].mean()
        currentList.append(abs(steadyStateForwardCurrent-steadyStateReverseCurrent)/2)

    if not trasatti:
        
        #plots ECSA CVs
        legendList = ['{:3.0f} '.format(i*1000)+r'$\frac{mV}{s}$' for i in scanRateList]
        plotCompareCV(dataList,legendList,title+' EDLC CVs',horizontalLine=True,currentdensity=False)
        
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
        plotCompareCV(dataList,legendList,title+' EDLC CVs',horizontalLine=True)
            
        #performs linear regression
        result = linregress(xList,yList)
        k1 = result.slope
        k2 = result.intercept
    
    return

def buildDataList(filenameList,pH,area,referencePotential): 
    
    dataList = []
    
    for filename in filenameList:
        data = readCV(filename,pH,area,referencePotential)
        dataList.append(data)
    
    return dataList

def buildEDLCList(folderName,number,pH,area,referencePotential,excludeLastX=0):
    
    twoDigit = False
    number = str(number)
    if len(number) == 2:
        twoDigit = True
    
    if not os.path.isdir(folderName):
        return None

    files = os.listdir(folderName)
    edlcFiles = []
    for file in files:
        isEDLC = False
        if twoDigit and (file[:2] == number):
            isEDLC = True
        elif (not twoDigit) and (file[0] == number):
            isEDLC = True
        if (file[-3:] != 'txt') and (file[-3:] != 'mpt'):
            isEDLC = False
        if 'CA' in file:
            isEDLC = False
        if isEDLC:
            edlcFiles.append(folderName + '\\' + file)
    
    if excludeLastX != 0:        
        edlcFiles = edlcFiles[excludeLastX:]

    return buildDataList(edlcFiles,pH,area,referencePotential)