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
    plt.tight_layout()
    plt.show()
    return (fig, ax)
    
def plotManyCVs(dataList: list[pd.DataFrame], title: str, legendList: list[str] = None, cycleList: list[int] = None, horizontalLine: bool = True, verticalLine: bool = True, xlim: list[float] = None, ylim: list[float] = None, currentdensity: bool = True, customColors: list[str] = None):
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
        customColors(list[str], optional): list of custom colors to use

    Returns:
        tuple(matplotlib.figure.Figure,list[matplotlib.axes._axes.Axes]): fig and ax for further customization if necessary
    """
    
    numFiles = len(dataList)
    
    fig, ax = plt.subplots()
    
    if horizontalLine:
        ax.axhline(0,color='k')
    if verticalLine:
        ax.axvline(0,color='k')
        ax.axvline(1233,color='k')
        
    for i in range(numFiles):
        
        data = dataList[i]
        
        if customColors == None:
            color = colorFader('blue','red',i,numFiles)
        else:
            color = customColors[i]
        
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
    plt.tight_layout()
    plt.show()
    return (fig, ax)

def plotECSA(dataList: list[pd.DataFrame], title: str, trasatti: bool = False, show=True):
    """Takes list of DataFrames from buildEDLCList and finds ECSA.

    Args:
        dataList (list[pd.Dataframe]): CV DataFrames
        title (str): title of graph + EDLC
        trasatti (bool): set to False, implements Trasatti's method, unfinished
        show (bool): set to True, whether to plot graphs or not

    Returns:
        tuple(list[float], list[float], LinregressResult | None):
            (scanRateList, currentList, linregress_result). Third item is
            None if fewer than 2 valid CVs were produced.
    """
    currentList = []
    scanRateList = []

    for cvIdx, data in enumerate(dataList):

        if data.empty:
            continue

        try:
            numCycles = int(data['cycle number'].max())
        except Exception:  # empty dataframe, missing column, etc.
            continue

        # take the last full cycle (second-to-last if >1, since the final
        # cycle often gets cut off)
        cycle = 1 if numCycles == 1 else numCycles - 1
        dataSlice = data[data['cycle number'] == cycle].copy()
        if dataSlice.empty:
            continue

        # scan rate from the positive-slope portion of Ewe(t)
        dataSlice['Scan Rate (V/s)'] = data['Ewe/V'].diff() / data['time/s'].diff()
        scanRate = dataSlice.loc[dataSlice['Scan Rate (V/s)'] > 0,
                                 'Scan Rate (V/s)'].mean()
        if not np.isfinite(scanRate):
            print(f"Warning: CV {cvIdx} has no positive-slope samples; skipping.")
            continue
        scanRateList.append(scanRate)

        # ---- find turning points in control/V ----------------------------
        voltage = dataSlice['control/V'].values
        if len(voltage) < 3:
            print(f"Warning: CV {cvIdx} too short ({len(voltage)} pts); skipping.")
            scanRateList.pop()
            continue

        voltage_diff = np.diff(voltage)

        # Forward-fill zero-sign samples: Biologic emits repeated samples on
        # staircase waveforms, and np.sign(0) = 0 causes phantom direction
        # flips at every flat step. Hold the last nonzero sign across them.
        signs = np.sign(voltage_diff)
        signs = (pd.Series(signs).replace(0, np.nan)
                 .ffill().bfill().fillna(0).values)

        sign_changes = np.diff(signs)
        turning_indices = np.where(sign_changes != 0)[0] + 1

        # Prominence filter: require excursion between candidate turns to
        # clear 5% of the total sweep range before counting it as real.
        voltage_range = voltage.max() - voltage.min()
        if voltage_range > 0:
            prominence = 0.05 * voltage_range
            filtered = []
            last_extremum = voltage[0]
            for idx in turning_indices:
                if abs(voltage[idx] - last_extremum) >= prominence:
                    filtered.append(idx)
                    last_extremum = voltage[idx]
            turning_indices = np.array(filtered, dtype=int)

        # argmin / argmax are always genuine turns — keep as safety net
        turning_indices = np.append(turning_indices,
                                    [np.argmin(voltage), np.argmax(voltage)])
        turning_indices = np.sort(np.unique(turning_indices))

        # ---- split into forward / reverse segments -----------------------
        forwardSlice = pd.DataFrame()
        reverseSlice = pd.DataFrame()

        going_up = voltage_diff[0] > 0

        start_idx = 0
        for j, turn_idx in enumerate(turning_indices):
            segment = dataSlice.iloc[start_idx:turn_idx + 1].copy()
            if len(segment) > 1:
                segment_diff = np.diff(segment['control/V'].values)
                segment_direction = np.mean(segment_diff) > 0
                if segment_direction == going_up:
                    forwardSlice = pd.concat([forwardSlice, segment], ignore_index=True)
                else:
                    reverseSlice = pd.concat([reverseSlice, segment], ignore_index=True)
            start_idx = turn_idx

        final_segment = dataSlice.iloc[start_idx:].copy()
        if len(final_segment) > 1:
            final_diff = np.diff(final_segment['control/V'].values)
            final_direction = np.mean(final_diff) > 0
            if final_direction == going_up:
                forwardSlice = pd.concat([forwardSlice, final_segment], ignore_index=True)
            else:
                reverseSlice = pd.concat([reverseSlice, final_segment], ignore_index=True)

        if forwardSlice.empty or reverseSlice.empty \
                or 'time/s' not in forwardSlice.columns \
                or 'time/s' not in reverseSlice.columns:
            print(f"Warning: CV {cvIdx} didn't split into both sweeps "
                  f"(forward={len(forwardSlice)}, reverse={len(reverseSlice)}); "
                  f"skipping.")
            scanRateList.pop()
            continue

        # ---- steady-state current near the end of each sweep -------------
        # iloc is safer than loc here since concat reindexes
        steadyStateForwardCurrent = forwardSlice['I/A'].iloc[-10:-1].mean()
        steadyStateReverseCurrent = reverseSlice['I/A'].iloc[-10:-1].mean()
        currentList.append(abs(steadyStateForwardCurrent - steadyStateReverseCurrent) / 2)

    # ---- bail if we don't have enough points to fit ----------------------
    if len(scanRateList) < 2 or len(currentList) < 2:
        print(f"Warning: only {len(scanRateList)} valid CV(s) for '{title}'; "
              f"cannot run linregress.")
        return (scanRateList, currentList, None)

    if not trasatti:

        legendList = ['{:3.0f} '.format(j * 1000) + r'$\frac{mV}{s}$'
                      for j in scanRateList]
        if show:
            plotManyCVs(dataList, title + ' EDLC CVs',
                        legendList=legendList,
                        horizontalLine=True,
                        currentdensity=False,
                        verticalLine=False)

        result = linregress(scanRateList, currentList)

        fig, ax = plt.subplots()
        ax.plot(scanRateList, currentList, 'ko')
        xValues = np.linspace(0, ax.get_xlim()[1], 3)
        yValues = (xValues * result.slope) + result.intercept
        ax.plot(xValues, yValues)
        ax.set(title=title + ' EDLC Plot, Slope = {:.5f} $\mu$F'.format(result.slope * 1e6),
               xlabel='Scan Rate (V/s)',
               ylabel='Current (A)',
               xlim=[0, ax.get_xlim()[1]],
               ylim=[0, ax.get_ylim()[1]])
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close(fig)

    else:  # trasatti branch — unfinished in original, preserved as-is
        xList = [sr ** 0.5 for sr in scanRateList]
        yList = [c / x for c, x in zip(currentList, xList)]
        legendList = ['{:3.0f} '.format(i * 1000) + r'$\frac{mV}{s}$'
                      for i in scanRateList]
        plotManyCVs(dataList, legendList, title + ' EDLC CVs', horizontalLine=True)
        result = linregress(xList, yList)

    return (scanRateList, currentList, result)

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
    
    #plt.tight_layout()
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
    
    plt.tight_layout()
    plt.show()

    return (fig, ax)