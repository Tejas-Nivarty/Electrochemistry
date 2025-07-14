import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from ReadDataFiles import colorFader

def plotOneCV(data: pd.DataFrame, title: str, legendDisplay: bool = True):
    """Plots every cycle of one CV scan.

    Args:
        data (pd.DataFrame): DataFrame from readCV.
        title (str): Title of plot.
        legendDisplay (bool, True): Whether to display legend, default True.

    Returns:
        tuple(matplotlib.figure.Figure,list[matplotlib.axes._axes.Axes]): fig and ax for further customization if necessary
    """
    numCycles = int(data['cycle number'].max())
    fig, ax = plt.subplots()
    ax.axhline(0,color='k')
    ax.axvline(0,color='k')
    for i in range(1,numCycles+1):
        initialOrFinal = False
        if i == 1 or i == numCycles:
            legendAddition = ''
        elif numCycles > 10:
            legendAddition = '_'
        else:
            legendAddition = ''
        color = colorFader('green','blue',i-1,numCycles)
        dataSlice = data[data['cycle number'] == i]
        ax.plot(dataSlice['Ewe/V'],dataSlice['j/mA*cm-2'],color = color,label = legendAddition+'Cycle '+str(i))
    ax.set(title = title,
           xlabel = r'$V_{RHE}$',
           ylabel = r'j $(\frac{mA}{cm^2_{geo}})$')
    ax.legend()
    plt.show()
    return (fig, ax)
    
def plotManyCVs(dataList: list[pd.DataFrame], title: str, legendList: list[str] = None, cycleList: list[int] = None, horizontalLine: bool = True, verticalLine: bool = True, xlim: list[float] = None, ylim: list[float] = None, currentdensity: bool = True):
    """Plots one cycle (default last) of many CVs.

    Args:
        dataList (list[pd.DataFrame]): List of CV DataFrames.
        title (str): Title of plot.
        legendList (list[str], optional): List of legend items. Defaults to None.
        cycleList (list[int], optional): List of specific cycles to plot. Defaults to None.
        horizontalLine (bool, optional): Plots line through current = 0. Defaults to True.
        verticalLine (bool, optional): Plots line through V_RHE = 0. Defaults to True.
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
        except Exception as e: #empty dataframe, no data
            print('exception')
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
            try:
                if currentdensity:
                    ax.plot(dataSlice['Ewe/mV'],dataSlice['j/mA*cm-2'],color = color,label = legendList[i])
                else:
                    ax.plot(dataSlice['Ewe/mV'],dataSlice['I/A'],color = color,label = legendList[i])
            except IndexError:
                continue
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

def plotECSA(dataList: list[pd.DataFrame], title: str, trasatti: bool = False, show = True):
    """Takes list of DataFrames from buildEDLCList and finds ECSA.

    Args:
        dataList (list[pd.Dataframe]): CV DataFrames
        title (str): title of graph + EDLC
        trasatti (bool): set to False, implements Trasatti's method, unfinished
        show (bool): set to True, whether to plot graphs or not

    Returns:
        tuple(matplotlib.figure.Figure,list[matplotlib.axes._axes.Axes]): fig and ax for further customization if necessary
    """
    
    currentList = []
    scanRateList = []
    
    for i, data in enumerate(dataList):
        
        try:
            numCycles = int(data['cycle number'].max())
        except Exception as e: #empty dataframe most likely
            continue
        
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
        
        # Solution that properly CONCATENATES forward and reverse segments
        voltage = dataSlice['control/V'].values
        time = dataSlice['time/s'].values

        # Find ALL turning points using derivative
        voltage_diff = np.diff(voltage)
        sign_changes = np.diff(np.sign(voltage_diff))
        turning_indices = np.where(sign_changes)[0] + 1

        # Also consider extrema
        turning_indices = np.append(turning_indices, [np.argmin(voltage), np.argmax(voltage)])
        turning_indices = np.sort(np.unique(turning_indices))

        # print(f"Found {len(turning_indices)} turning points at: {turning_indices}")

        # Create empty DataFrames for concatenated segments
        forwardSlice = pd.DataFrame()
        reverseSlice = pd.DataFrame()

        # Determine sweep direction
        going_up = voltage_diff[0] > 0

        # Loop through segments and concatenate to appropriate slice
        start_idx = 0
        for i, turn_idx in enumerate(turning_indices):
            # Get current segment
            segment = dataSlice.iloc[start_idx:turn_idx+1].copy()
            
            # Check segment direction (using average derivative)
            if len(segment) > 1:
                segment_diff = np.diff(segment['control/V'].values)
                segment_direction = np.mean(segment_diff) > 0
                
                # Add segment to appropriate slice based on direction
                if segment_direction == going_up:
                    # This is a forward segment
                    forwardSlice = pd.concat([forwardSlice, segment], ignore_index=True)
                else:
                    # This is a reverse segment
                    reverseSlice = pd.concat([reverseSlice, segment], ignore_index=True)
            
            # Next segment starts at this turning point
            start_idx = turn_idx
            
        # Add final segment
        final_segment = dataSlice.iloc[start_idx:].copy()
        if len(final_segment) > 1:
            final_diff = np.diff(final_segment['control/V'].values)
            final_direction = np.mean(final_diff) > 0
            
            if final_direction == going_up:
                forwardSlice = pd.concat([forwardSlice, final_segment], ignore_index=True)
            else:
                reverseSlice = pd.concat([reverseSlice, final_segment], ignore_index=True)

        # # Verify the concatenation worked
        # print(f"Forward slice: {len(forwardSlice)} points")
        # print(f"Reverse slice: {len(reverseSlice)} points")

        # # Debug info
        # print(f"Split at index {turn_idx}/{len(voltage)}")
        # print(f"Forward scan: {len(forwardSlice)} points, {voltage[0]:.3f}V → {voltage[turn_idx]:.3f}V")
        # print(f"Reverse scan: {len(reverseSlice)} points, {voltage[turn_idx]:.3f}V → {voltage[-1]:.3f}V")
        
        # plt.plot(forwardSlice['time/s'],forwardSlice['control/V'],'k')
        # plt.plot(reverseSlice['time/s'],reverseSlice['control/V'],'r--')
        # plt.show()
        
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
        if show:
            plotManyCVs(dataList,title+' EDLC CVs',legendList=legendList,horizontalLine=True,currentdensity=False,verticalLine=False)
        
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
        if show:
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
    
    if not show:
        plt.close(fig)
    
    return (scanRateList,currentList,result)

def plotManyECSAs(
        dataSets: list[list[pd.DataFrame]],
        title: str = "",
        labels: list[str] | None = None
    ):
    """
    Overlay multiple ECSA-vs-scan-rate regressions in one figure.

    Parameters
    ----------
    dataSets : list[list[pd.DataFrame]]
        Outer list = one sample / condition.
        Inner list = the CV DataFrames for that sample (exactly what plotECSA expects).
    title : str
        Figure title (gets "EDLC Plot" appended automatically).
    labels : list[str] | None
        Legend labels, one per dataset. Defaults to «Sample 0», «Sample 1», …

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """
    if labels is None:
        labels = [f"EDLC {i}" for i in range(len(dataSets))]
    if len(labels) != len(dataSets):
        raise ValueError("labels and dataSets must be the same length")

    fig, ax = plt.subplots()
    slopes_uF = []
    linregressList = []
    
    totalNumber = len(dataSets)

    for idx, (dataList, label) in enumerate(zip(dataSets, labels)):
        scanRateList, currentList, linregress = plotECSA(dataList,'',show=False)

        if not currentList or not scanRateList:
            continue  # empty dataset → skip

        linregressList.append(linregress)
        # linear regression
        slope_uF = linregress.slope * 1e6  # convert to µF
        slopes_uF.append(slope_uF)

        # plotting
        c = colorFader('blue','red',idx,totalNumber)
        ax.plot(scanRateList, currentList, "o", color=c, label=f"{label} ({slope_uF:.2f} µF)")
        
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()
        
    for idx, (dataList, label) in enumerate(zip(dataSets,labels)):
        
        x_line = np.linspace(0, xlims[1], 3)
        c = colorFader('blue','red',idx,totalNumber)
        ax.plot(x_line,
                linregressList[idx].slope * x_line + linregressList[idx].intercept,
                "-", color=c, label='_')
        ax.set(ylim = ylims,
               xlim = xlims)

    ax.set(
        title=f"{title} EDLC Plot" if title else "EDLC Plot",
        xlabel="Scan Rate (V/s)",
        ylabel="Current (A)",
        xlim=(0, ax.get_xlim()[1]),
        ylim=(0, ax.get_ylim()[1]),
    )
    ax.legend(fontsize=8)
    
    #plt.show()
    
    
    fig, ax = plt.subplots()
    xLabels = np.arange(1,len(slopes_uF)+1,1)
    yLabels = slopes_uF
    ax.plot(xLabels,yLabels,
            color='k',
            linestyle='--',
            marker='o')
    
    ax.set(title=title,
           xlabel='Experiment Number',
           ylabel='ECSA ($\mu$F)')
    
    plt.show()

    return (fig, ax)