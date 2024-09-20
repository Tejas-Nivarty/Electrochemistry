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
        color = colorFader('green','blue',i/numCycles)
        dataSlice = data[data['cycle number'] == i]
        ax.plot(dataSlice['Ewe/V'],dataSlice['j/mA*cm-2'],color = color,label = 'Cycle '+str(i))
    ax.set(title = title,
           xlabel = r'$V_{RHE}$',
           ylabel = r'j $(\frac{mA}{cm^2_{geo}})$')
    ax.legend()
    plt.show()
    return
    
def plotCompareCV(dataList,legendList,title,cycleList=None,horizontalLine=False,verticalLine=False,xlim=None,ylim=None):
    numFiles = len(dataList)
    fig, ax = plt.subplots()
    if horizontalLine:
        ax.axhline(0,color='k')
    if verticalLine:
        ax.axvline(0,color='k')
    for i in range(numFiles):
        data = dataList[i]
        color = colorFader('blue','red',i/numFiles)
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
            ax.plot(dataSlice['Ewe/mV'],dataSlice['j/mA*cm-2'],color = color)
        else:
            ax.plot(dataSlice['Ewe/mV'],dataSlice['j/mA*cm-2'],color = color,label = legendList[i])
        
    ax.set(title = title,
           xlabel = r'$mV_{RHE}$',
           ylabel = r'j $(\frac{mA}{cm^2_{geo}})$')
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
        
        #splits last cycle into oxidative and reductive currents
        oxidativeCurrentSlice = dataSlice[dataSlice['ox/red'] == 1]
        reductiveCurrentSlice = dataSlice[dataSlice['ox/red'] == 0]
        #finds maximum oxidative and maximum reductive current
        maxOxidativeCurrent = oxidativeCurrentSlice['I/A'].abs().max()
        maxReductiveCurrent = reductiveCurrentSlice['I/A'].abs().max()
        #takes smaller of the peak oxidative and reductive currents (less faradaic contribution)
        if maxOxidativeCurrent > maxReductiveCurrent:
            currentList.append(maxReductiveCurrent)
        else:
            currentList.append(maxOxidativeCurrent)

    if not trasatti:
        
        #plots ECSA CVs
        legendList = ['{:3.0f} '.format(i*1000)+r'$\frac{mV}{s}$' for i in scanRateList]
        plotCompareCV(dataList,legendList,title+' EDLC CVs',horizontalLine=True)
        
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
        if file[-3:] != 'txt':
            isEDLC = False
        if isEDLC:
            edlcFiles.append(folderName + '\\' + file)
    
    if excludeLastX != 0:        
        edlcFiles = edlcFiles[excludeLastX:]

    return buildDataList(edlcFiles,pH,area,referencePotential)

# pre = buildEDLCList(r'Joana Data\2024-07-30-JD-01-011',3,14,0.1826403875,0.209,excludeLastX=1)
# post = buildEDLCList(r'Joana Data\2024-07-30-JD-01-011',13,14,0.1826403875,0.209,excludeLastX=1)
# plotECSA(pre,'Pre')
# plotECSA(post,'Post')

# #readCV, plotCompareCV

# post = readCV(r'Data_Files\2024-06-18-TN-01-044\15_CV_02_CV_C02.txt',
#               14,
#               0.063253258)

# pre = readCV(r'Data_Files\2024-05-01-TN-01-038\12_CV_02_CV_C02.txt',
#              14,
#              0.063253258)

# plotCompareCV([pre,post],['Pre-','Post-'],'20 nm SRO Underlayer Ar Purged',
#               horizontalLine=True,
#               verticalLine=True)