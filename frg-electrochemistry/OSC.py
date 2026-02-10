import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import re
from pathlib import Path
plt.rcParams['font.size'] = 12
import matplotlib as mpl
from ReadDataFiles import readOSC, colorFader, calculateIntegral
import argparse
from bokeh.plotting import figure, show
from bokeh.layouts import column, row
from bokeh.models import Range1d, LinearAxis
from bokeh.io import output_notebook, reset_output
from bokeh.palettes import Viridis256, Spectral11
import datashader as ds
import datashader.transfer_functions as tf
InteractiveImage = None
from colorcet import fire
import holoviews as hv
hv.extension('bokeh') 
from holoviews.operation.datashader import datashade, shade

def fft(data: pd.DataFrame,dataLabel: str,timeLabel: str):
    """Finds the magnitudes from a fast Fourier transform of a dataframe with x and y values.

    Args:
        data (pd.DataFrame): DataFrame with all data.
        dataLabel (str): Label for DataFrame y-values (current, voltage, etc.)
        timeLabel (str): Label for DataFrame x-values (typically time)

    Returns:
        tuple[np.ndarray,np.ndarray]: (frequency, magnitude)
    """
    dt = data[timeLabel].diff().mean()
    
    data[dataLabel] = abs(data[dataLabel] - data[dataLabel].mean())
    
    inputArray = np.array(data[dataLabel])
    #inputArray = np.tile(inputArray, 20)
    L = len(inputArray)
    
    fftValues = sc.fft.rfft(inputArray)

    magnitude = np.abs(fftValues/L)
    magnitude = magnitude[:L // 2]*2
    
    frequencies = sc.fft.fftfreq(len(inputArray), dt)
    frequencies = frequencies[:L // 2]
    
    return (frequencies,magnitude)

def plotFFTs(datasets: list[tuple], legend: list[str], title: str):
    """Takes FFT from OSC.fft and plots many of them.

    Args:
        datasets (list[tuple[np.ndarray,np.ndarray]]): List of OSC.fft outputs.
        legend (list[str]): Legend text for each dataset in datasets.
        title (str)): Title of plot.
    """
    fig, ax = plt.subplots()
    maxMagnitude = 0
    
    #replace with datashader
    for i, data in enumerate(datasets):
        
        color = colorFader('blue','red',i,len(datasets))
        frequencies = data[0]
        magnitude = data[1]
        if maxMagnitude < max(magnitude):
            maxMagnitude = max(magnitude)
        ax.scatter(frequencies,magnitude,
                   color=color,
                   s=4,
                   label=legend[i])
    
    ax.set(title = title,
           xlabel = 'Frequency (Hz)',
           ylabel = 'Magnitude',
           xscale = 'log',
           #xlim = [10**3,125000000/2],
           yscale = 'log',
           #ylim = [10**-6,1]
           )
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return

def analyzeWaveform(pulse: pd.DataFrame, experimentLength: float, frequency: float, title: str = None, plot: bool = True):
    """Integrates oscilloscope waveform to find total charge transferred. Doesn't work well.

    Args:
        pulse (pd.DataFrame): Oscilloscope waveform from ReadDataFiles.readOSC()
        experimentLength (float): Length of experiment that waveform was collected over in seconds.
        frequency (float): Frequency of waveform in Hz.
        title (str, optional): Title of sample to be included in plots. Defaults to None.
        plot (bool, optional): Whether to plot waveform. Defaults to True.
    """
    numberOfWaveforms = pulse['Time (s)'].max()*frequency
    period = 1/frequency
    
    upPulseStart = 0
    upPulseEnd = period/2
    dnPulseStart = period/2
    dnPulseEnd = period
    
    if numberOfWaveforms >= 3:
        upPulseStart = period
        upPulseEnd = 1.5*period
        dnPulseStart = 1.5*period
        dnPulseEnd = 2*period
        
    elif numberOfWaveforms >= 2:
        upPulseStart = period
        upPulseEnd = 1.5*period
        dnPulseStart = 0.5*period
        dnPulseEnd = period
        
    
    upPulseCharge = calculateIntegral(pulse['Time (s)'],
                                      pulse['Current (A)'],
                                      0,
                                      [upPulseStart,upPulseEnd])
    dnPulseCharge = calculateIntegral(pulse['Time (s)'],
                                      pulse['Current (A)'],
                                      0,
                                      [dnPulseStart,dnPulseEnd])
    
    ferroelectricChargeEstimate = (upPulseCharge - dnPulseCharge)/2
    print('Charge Transferred to Ferroelectric During One Switching Pulse: {} C'.format(ferroelectricChargeEstimate))
    
    totalChargeTransferred = upPulseCharge + dnPulseCharge
    print('Charge Transferred During One Full Waveform: {} C'.format(totalChargeTransferred))
    
    totalChargeTransferred *= experimentLength/period
    print('Total Charge Transferred: {} C'.format(totalChargeTransferred))
    totalChargeTransferred = pulse['Charge (C)'].iloc[-1]*experimentLength*frequency
    print('Total Charge Transferred: {} C'.format(totalChargeTransferred))
    
    
    if plot:
        plotWaveform(pulse,title,jv=True)

    return

def plotWaveformMPL(pulse: pd.DataFrame, title: str, jv: bool, currentDensity: bool = False, reference: pd.DataFrame = pd.DataFrame()):
    """Uses matplotlib to plot waveform. Slow for low frequency, large waveforms.

    Args:
        pulse (pd.DataFrame): Waveform from readOSC.
        title (str): Title of plot.
        jv (bool): If true, plots jv curve.
        currentDensity (bool): If true, uses current density rather than raw current
        reference (pd.DataFrame, optional): Reference from readRawWaveform. Defaults to empty pd.DataFrame().

    Returns:
        tuple(matplotlib.figure.Figure,matplotlib.axes._axes.Axes): fig and ax for further customization if necessary
    """
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    if currentDensity:
        ax.plot(pulse['Time (s)']*1000,pulse['Current Density (mA/cm^2)'],color='r')
        ylabel = r'Current Density $(\frac{mA}{cm^2_{geo}})$'
    else:
        ax.plot(pulse['Time (s)']*1000,pulse['Current (A)'],color='r')
        ylabel=r'Current (A)'
    ax2.plot(pulse['Time (s)']*1000,pulse['Voltage (V)'],color='k')
    if not reference.empty:
        ax2.plot(reference['Time (s)']*1000,reference['Voltage (V)'],color='k',linestyle='--')
    ax.set(title= title,
           ylabel=ylabel,
           xlabel='Time (ms)')
    ax2.set(ylabel = r'Voltage (V$_{RHE}$)')
    ax.axhline(0,color='k',zorder=0)
    plt.tight_layout()
    plt.show()
    
    if jv == True:
        fig, ax = plt.subplots()
        plt.axhline(0,color='k',zorder=0)
        plt.axvline(0,color='k',zorder=0)
        # plt.axvline(-1.965,color='k',linestyle='--',zorder=0)
        # plt.axvline(6.535,color='k',linestyle='--',zorder=0)
        # plt.axvline(-0.335,color='k',linestyle='--',zorder=0)
        plt.scatter(pulse['Voltage (V)'],pulse['Charge Density (mC/cm^2)'],c=(pulse['Time (s)']-pulse['Time (s)'].iloc[0])*1e6,cmap='gist_rainbow')
        #plt.scatter(pulse['Voltage (V)'],pulse['Current Density (mA/cm^2)'],c=(pulse['Time (s)']-pulse['Time (s)'].iloc[0])*1e6,cmap='gist_rainbow')
        plt.colorbar(label=r'Time ($\mu$s)')
        plt.title(title)
        plt.xlabel(r'Voltage (V$_{RHE}$)')
        plt.ylabel(r'Charge Density $(\frac{mC}{cm^2_{geo}})$')
        #plt.ylabel(r'Current Density $(\frac{mA}{cm^2_{geo}})$')
        plt.tight_layout()
        plt.show()
    
    return (fig, ax)

def plotWaveformsMPL(pulses: list[pd.DataFrame], title: str, legend: list[str], jv: bool, reference: pd.DataFrame = pd.DataFrame(), customColors: list = None):
    """Plots multiple waveforms using matplotlib.

    Args:
        pulses (list[pd.DataFrame]): List of waveforms from readOSC.
        title (str): Title of plots.
        legend (list[str]): Legends of pulses.
        jv (bool): If true, plots jv curves.
        reference (pd.DataFrame, optional): Reference waveform from readRawWaveform. Defaults to pd.DataFrame().
        customColors (list, optional): List of custom colors. Defaults to None.

    Returns:
        tuple(matplotlib.figure.Figure,matplotlib.axes._axes.Axes): fig and ax for further customization if necessary
    """
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    
    
    
    for i, pulse in enumerate(pulses):
        if customColors == None:
            color = colorFader('blue','red',i,len(pulses))
        else:
            color = customColors[i]
        ax.plot(pulse['Time (s)']*1000,pulse['Current Density (mA/cm^2)'],color=color)
        ax2.plot(pulse['Time (s)']*1000,pulse['Voltage (V)'],color=color,linestyle=':')
        
    if not reference.empty:
        ax2.plot(reference['Time (s)']*1000,reference['Voltage (V)'],color='k',linestyle='--')
    
    ax.set(title= title,
        ylabel=r'Current Density $(\frac{mA}{cm^2_{geo}})$',
        xlabel='Time (ms)')
    ax.legend(legend)
    ax2.set(ylabel = r'Voltage (V$_{RHE}$)')
    ax.axhline(0,color='k',zorder=0)
    plt.tight_layout()
    plt.show()
    
    if jv == True:
        fig, ax = plt.subplots()
        
        for i, pulse in enumerate(pulses):
            if customColors == None:
                color = colorFader('blue','red',i,len(pulses))
            else:
                color = customColors[i]
            ax.plot(pulse['Voltage (V)'],pulse['Current Density (mA/cm^2)'],color=color)
        ax.legend(legend)
        ax.set(title = title,
               ylabel = r'Current Density $(\frac{mA}{cm^2_{geo}})$',
               xlabel = 'Voltage (V)')
        
        ax.axhline(0,color='k',zorder=0)
        ax.axvline(0,color='k',zorder=0)
        ax.axvline(-1.965,color='k',linestyle='--',zorder=0)
        ax.axvline(6.535,color='k',linestyle='--',zorder=0)
        ax.axvline(-0.335,color='k',linestyle='--',zorder=0)
        plt.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots()
        
        for i, pulse in enumerate(pulses):
            if customColors == None:
                color = colorFader('blue','red',i,len(pulses))
            else:
                color = customColors[i]
            ax.plot(pulse['Voltage (V)'],pulse['Charge Density (mC/cm^2)'],color=color)
        ax.legend(legend)
        ax.set(title = title,
               ylabel = r'Charge Density $(\frac{mC}{cm^2_{geo}})$',
               xlabel = 'Voltage (V)')
        ax.axhline(0,color='k',zorder=0)
        ax.axvline(0,color='k',zorder=0)
        ax.axvline(-1.965,color='k',linestyle='--',zorder=0)
        ax.axvline(6.535,color='k',linestyle='--',zorder=0)
        ax.axvline(-0.335,color='k',linestyle='--',zorder=0)
        plt.tight_layout()
        plt.show()
    
    return (fig, ax)

def plotWaveform(pulse: pd.DataFrame, title: str, jv: bool, reference=pd.DataFrame()):
    """
    Plot waveform data using Bokeh with improved formatting. From Claude.
    
    Parameters:
    -----------
    pulse : pd.DataFrame
        Waveform data
    title : str
        Plot title
    jv : bool
        Whether to plot current-voltage relationship
    reference : pd.DataFrame, optional
        Reference data to overlay
    
    Returns:
    --------
    tuple
        (time_voltage_current_plot, jv_plot) if jv=True, else time_voltage_current_plot
    """
    # Import required modules
    from bokeh.plotting import figure, show
    from bokeh.models import Range1d, LinearAxis, ColumnDataSource, Label
    from bokeh.layouts import column
    from bokeh.models import ColorBar, LinearColorMapper
    import pandas as pd
    import numpy as np
    
    # Convert time to ms for plotting
    time_ms = pulse['Time (s)'] * 1000
    
    # Time-Current-Voltage plot with adjusted margins
    p1 = figure(title=title, 
                x_axis_label='Time (ms)', 
                y_axis_label='Current (A)',
                width=800, 
                height=400,
                tools='pan,wheel_zoom,box_zoom,reset,save',
                min_border_left=50,  # Add more space on the left for y-axis
                min_border_right=50)  # Add more space on the right for second y-axis
    
    # Calculate appropriate y-range for current to ensure variations are visible
    # Get the non-zero current values to compute a better scale
    current_values = pulse['Current (A)'].values
    non_zero_current = current_values[np.abs(current_values) > 1e-6]
    
    if len(non_zero_current) > 0:
        # Set y-range based on the actual data range with padding
        current_min = np.min(non_zero_current) * 1.2 if np.min(non_zero_current) < 0 else np.min(non_zero_current) * 0.8
        current_max = np.max(non_zero_current) * 1.2 if np.max(non_zero_current) > 0 else np.max(non_zero_current) * 0.8
        
        # Ensure we don't zoom in too much on near-zero values
        if abs(current_max - current_min) < 0.1 * np.max(np.abs(non_zero_current)):
            margin = 0.1 * np.max(np.abs(non_zero_current))
            current_min = min(current_min, -margin)
            current_max = max(current_max, margin)
        
        p1.y_range = Range1d(current_min, current_max)
    
    # Add current plot to left y-axis
    p1.line(time_ms, pulse['Current (A)'], line_width=2, color='red', legend_label='Current')
    
    # Create a second y-axis for voltage with appropriate scaling
    # Calculate range to highlight variations
    voltage_values = pulse['Voltage (V)'].values
    voltage_range = voltage_values.max() - voltage_values.min()
    
    # Add a 10% margin on each side
    voltage_min = voltage_values.min() - 0.1 * voltage_range
    voltage_max = voltage_values.max() + 0.1 * voltage_range
    
    # If range is very small, add more padding to make variations visible
    if voltage_range < 0.1 * np.max(np.abs(voltage_values)):
        padding = 0.1 * np.max(np.abs(voltage_values))
        voltage_min = voltage_values.min() - padding
        voltage_max = voltage_values.max() + padding
    
    p1.extra_y_ranges = {"voltage": Range1d(start=voltage_min, end=voltage_max)}
    
    # Add a second axis to the right
    voltage_axis = LinearAxis(y_range_name="voltage", axis_label=r"Voltage $(V_{RHE})$")
    p1.add_layout(voltage_axis, 'right')
    
    # Add voltage line on right axis
    p1.line(time_ms, pulse['Voltage (V)'], 
            line_width=2, color='black', y_range_name="voltage", 
            legend_label='Voltage')
    
    # Add reference if provided
    if not reference.empty:
        ref_time_ms = reference['Time (s)'] * 1000
        p1.line(ref_time_ms, reference['Voltage (V)'], 
                line_width=2, color='black', y_range_name="voltage", 
                line_dash='dashed', legend_label='Reference Voltage')
    
    # Add zero line
    p1.line([time_ms.min(), time_ms.max()], [0, 0], line_width=1, color='black', alpha=0.5)
    
    # Configure legend
    p1.legend.location = "top_right"
    p1.legend.click_policy = "hide"
    
    # Fix axis alignment issues
    p1.yaxis.major_label_text_align = 'right'
    p1.yaxis.major_label_standoff = 10
    
    # Add some margin to make sure axes are fully visible
    p1.min_border_left = 50
    p1.min_border_right = 50
    
    # Apply theme settings for better appearance
    p1.xgrid.grid_line_color = 'lightgray'
    p1.ygrid.grid_line_color = 'lightgray'
    p1.xgrid.grid_line_alpha = 0.7
    p1.ygrid.grid_line_alpha = 0.7
    
    # Ensure axis labels don't get cut off
    p1.xaxis.axis_label_standoff = 12
    p1.yaxis.axis_label_standoff = 12
    
    # Show the first plot
    show(p1)
    
    # Create JV plot if requested
    if jv:
        # Create source for scatter plot with time as color dimension
        time_us = (pulse['Time (s)'] - pulse['Time (s)'].iloc[0]) * 1e6  # microseconds
        
        # Create a color mapper for time
        color_mapper = LinearColorMapper(palette='Viridis256', 
                                         low=time_us.min(), 
                                         high=time_us.max())
        
        # JV scatter plot with improved formatting
        p2 = figure(title=f"{title} - Current-Voltage Relationship", 
                    x_axis_label=r'Voltage $(V_{RHE})$', 
                    y_axis_label=r'Current Density $(\frac{mA}{cm^2_{geo}})$',
                    width=800, 
                    height=400,
                    tools='pan,wheel_zoom,box_zoom,reset,save,hover',
                    min_border_left=50,
                    min_border_right=50)
        
        # For large datasets, use direct scatter with downsampling
        if len(pulse) > 10000:
            # Downsample for better performance
            downsample_factor = max(1, len(pulse) // 10000)
            
            # Create source with downsampled data
            source = ColumnDataSource(data=dict(
                x=pulse['Voltage (V)'].iloc[::downsample_factor],
                y=pulse['Current Density (mA/cm^2)'].iloc[::downsample_factor],
                time=time_us.iloc[::downsample_factor]
            ))
        else:
            # For smaller datasets, use regular scatter with colormapping
            source = ColumnDataSource(data=dict(
                x=pulse['Voltage (V)'],
                y=pulse['Current Density (mA/cm^2)'],
                time=time_us
            ))
        
        # Add scatter points
        scatter = p2.scatter(x='x', y='y', source=source, 
                           size=5, fill_color={'field': 'time', 'transform': color_mapper},
                           line_color=None, alpha=0.7)
        
        # Add a color bar with proper formatting
        color_bar = ColorBar(color_mapper=color_mapper, 
                             label_standoff=12, 
                             border_line_color=None, 
                             location=(0, 0),
                             title="Time (μs)")
        p2.add_layout(color_bar, 'right')
        
        # Add reference lines
        p2.line([0, 0], [pulse['Current Density (mA/cm^2)'].min(), pulse['Current Density (mA/cm^2)'].max()], 
                line_width=1, color='black')
        p2.line([pulse['Voltage (V)'].min(), pulse['Voltage (V)'].max()], [0, 0], 
                line_width=1, color='black')
        
        # Add vertical reference lines
        for v in [-1.965, 6.535, -0.335]:
            p2.line([v, v], [pulse['Current Density (mA/cm^2)'].min(), pulse['Current Density (mA/cm^2)'].max()], 
                    line_width=1, color='black', line_dash='dashed')
        
        # Apply the same styling improvements
        p2.xgrid.grid_line_color = 'lightgray'
        p2.ygrid.grid_line_color = 'lightgray'
        p2.xgrid.grid_line_alpha = 0.7
        p2.ygrid.grid_line_alpha = 0.7
        p2.xaxis.axis_label_standoff = 12
        p2.yaxis.axis_label_standoff = 12
        
        # Create charge density plot with improved formatting
        p3 = figure(title=f"{title} - Charge-Voltage Relationship", 
                    x_axis_label=r'Voltage $(V_{RHE})$', 
                    y_axis_label=r'Charge Density $(\frac{mC}{cm^2_{geo}})$',
                    width=800, 
                    height=400,
                    tools='pan,wheel_zoom,box_zoom,reset,save,hover',
                    min_border_left=50,
                    min_border_right=50)
        
        # For large datasets, use downsampling for better performance
        if len(pulse) > 10000:
            # Downsample for better performance
            downsample_factor = max(1, len(pulse) // 10000)
            
            # Create source with downsampled data
            charge_source = ColumnDataSource(data=dict(
                x=pulse['Voltage (V)'].iloc[::downsample_factor],
                y=pulse['Charge Density (mC/cm^2)'].iloc[::downsample_factor],
                time=time_us.iloc[::downsample_factor]
            ))
        else:
            # For smaller datasets, use regular scatter with colormapping
            charge_source = ColumnDataSource(data=dict(
                x=pulse['Voltage (V)'],
                y=pulse['Charge Density (mC/cm^2)'],
                time=time_us
            ))
        
        # Add scatter points
        charge_scatter = p3.scatter(x='x', y='y', source=charge_source, 
                                  size=5, fill_color={'field': 'time', 'transform': color_mapper},
                                  line_color=None, alpha=0.7)
        
        # Add a color bar with proper formatting
        charge_color_bar = ColorBar(color_mapper=color_mapper, 
                                   label_standoff=12, 
                                   border_line_color=None, 
                                   location=(0, 0),
                                   title="Time (μs)")
        p3.add_layout(charge_color_bar, 'right')
        
        # Add reference lines
        p3.line([0, 0], [pulse['Charge Density (mC/cm^2)'].min(), pulse['Charge Density (mC/cm^2)'].max()], 
                line_width=1, color='black')
        p3.line([pulse['Voltage (V)'].min(), pulse['Voltage (V)'].max()], [0, 0], 
                line_width=1, color='black')
        
        # Add vertical reference lines
        for v in [-1.965, 6.535, -0.335]:
            p3.line([v, v], [pulse['Charge Density (mC/cm^2)'].min(), pulse['Charge Density (mC/cm^2)'].max()], 
                    line_width=1, color='black', line_dash='dashed')
        
        # Apply the same styling improvements
        p3.xgrid.grid_line_color = 'lightgray'
        p3.ygrid.grid_line_color = 'lightgray'
        p3.xgrid.grid_line_alpha = 0.7
        p3.ygrid.grid_line_alpha = 0.7
        p3.xaxis.axis_label_standoff = 12
        p3.yaxis.axis_label_standoff = 12
        
        # Show the plots in a column layout with proper spacing
        layout = column(p2, p3, spacing=20)
        show(layout)
        
        return p1, p2, p3
    
    return p1

def plotWaveforms(pulses: list[pd.DataFrame], title: str, legend: list[str], jv: bool, reference=pd.DataFrame(), customColors=None):
    """
    Plot multiple waveforms using Bokeh and Datashader for performance. From Claude.
    
    Parameters:
    -----------
    pulses : list[pd.DataFrame]
        List of waveform DataFrames
    title : str
        Plot title
    legend : list[str]
        Labels for each waveform
    jv : bool
        Whether to plot current-voltage relationship
    reference : pd.DataFrame, optional
        Reference data to overlay
    customColors : list, optional
        Custom colors for each waveform
    
    Returns:
    --------
    tuple
        (time_voltage_current_plot, jv_plot, charge_plot) if jv=True, else time_voltage_current_plot
    """
    # Create color palette
    if customColors is None:
        colors = [colorFader('blue', 'red', i, len(pulses)) for i in range(len(pulses))]
    else:
        colors = customColors
    
    # Optimize data for plotting
    # Precompute min and max values to avoid multiple loops through data
    min_time = min([pulse['Time (s)'].min() for pulse in pulses]) * 1000
    max_time = max([pulse['Time (s)'].max() for pulse in pulses]) * 1000
    min_voltage = min([pulse['Voltage (V)'].min() for pulse in pulses])
    max_voltage = max([pulse['Voltage (V)'].max() for pulse in pulses])
    min_current = min([pulse['Current Density (mA/cm^2)'].min() for pulse in pulses])
    max_current = max([pulse['Current Density (mA/cm^2)'].max() for pulse in pulses])
    
    # Create first plot (Time-Current-Voltage)
    # Determine if any dataset is large enough to benefit from datashader
    any_large_dataset = any(len(pulse) > 5000 for pulse in pulses)
    
    # Time series plots often don't need datashader as much as scatter plots
    # So we'll use regular Bokeh line plots here for better legend integration
    p1 = figure(title=title, 
                x_axis_label='Time (ms)', 
                y_axis_label='Current Density (mA/cm²geo)',
                width=800, 
                height=400,
                tools='pan,wheel_zoom,box_zoom,reset,save')
    
    # Create a second y-axis for voltage
    p1.extra_y_ranges = {"voltage": Range1d(
        start=min_voltage * 1.1,
        end=max_voltage * 1.1
    )}
    
    # Add a second axis to the right
    voltage_axis = p1.add_layout(
        LinearAxis(y_range_name="voltage", axis_label="Voltage (VRHE)"), 'right')
    
    # Add each waveform to the plot - use thinning for large datasets
    for i, pulse in enumerate(pulses):
        # Apply thinning for large datasets to improve performance
        if len(pulse) > 10000:
            # Take every Nth point where N scales with dataset size
            step = len(pulse) // 5000  # Target ~5000 points after thinning
            step = max(1, step)  # Ensure step is at least 1
            thinned_pulse = pulse.iloc[::step].copy()
        else:
            thinned_pulse = pulse
            
        time_ms = thinned_pulse['Time (s)'] * 1000
        
        # Add current plot to left y-axis
        p1.line(time_ms, thinned_pulse['Current Density (mA/cm^2)'], 
                line_width=2, color=colors[i], legend_label=legend[i])
        
        # Add voltage line on right axis (thinner line for less visual clutter)
        p1.line(time_ms, thinned_pulse['Voltage (V)'], 
                line_width=1, color=colors[i], y_range_name="voltage", 
                line_dash='dotted')
    
    # Add reference if provided
    if not reference.empty:
        ref_time_ms = reference['Time (s)'] * 1000
        p1.line(ref_time_ms, reference['Voltage (V)'], 
                line_width=2, color='black', y_range_name="voltage", 
                line_dash='dashed', legend_label='Reference Voltage')
    
    # Add zero line
    p1.line([min_time, max_time], [0, 0], line_width=1, color='black', alpha=0.5)
    
    # Configure legend
    p1.legend.location = "top_right"
    p1.legend.click_policy = "hide"
    
    # Show the first plot
    show(p1)
    
    # Create JV plots if requested
    if jv:
        # For JV plots, we'll use datashader for large datasets
        if any_large_dataset:
            # Combine all datasets for datashader with a source column
            combined_df = pd.DataFrame()
            for i, pulse in enumerate(pulses):
                temp_df = pd.DataFrame({
                    'voltage': pulse['Voltage (V)'],
                    'current': pulse['Current Density (mA/cm^2)'],
                    'charge': pulse['Charge Density (mC/cm^2)'],
                    'source': i  # Add source index for coloring
                })
                combined_df = pd.concat([combined_df, temp_df])
            
            # Create HoloViews plot for JV relationship
            points = hv.Points(combined_df, kdims=['voltage', 'current'], vdims=['source'])
            
            # Color by source
            color_points = hv.operation.datashader.datashade(points, aggregator=ds.count_cat('source'), 
                                                            color_key={i: c for i, c in enumerate(colors)})
            
            # Convert to Bokeh
            p2 = hv.render(color_points.opts(
                frame_width=800, frame_height=400,
                title=f"{title} - Current-Voltage Relationship",
                xlabel='Voltage (VRHE)',
                ylabel='Current Density (mA/cm²geo)',
                tools=['hover', 'box_zoom', 'wheel_zoom', 'pan', 'reset', 'save']
            ))
            
            # Create HoloViews plot for Charge-Voltage relationship
            charge_points = hv.Points(combined_df, kdims=['voltage', 'charge'], vdims=['source'])
            
            # Color by source
            color_charge_points = hv.operation.datashader.datashade(charge_points, aggregator=ds.count_cat('source'), 
                                                                   color_key={i: c for i, c in enumerate(colors)})
            
            # Convert to Bokeh
            p3 = hv.render(color_charge_points.opts(
                frame_width=800, frame_height=400,
                title=f"{title} - Charge-Voltage Relationship",
                xlabel='Voltage (VRHE)',
                ylabel='Charge Density (mC/cm²geo)',
                tools=['hover', 'box_zoom', 'wheel_zoom', 'pan', 'reset', 'save']
            ))
            
            # Add legend manually since datashader doesn't support it directly
            for i, label in enumerate(legend):
                p2.circle(x=[None], y=[None], color=colors[i], legend_label=label)
                p3.circle(x=[None], y=[None], color=colors[i], legend_label=label)
                
            # Configure legends
            p2.legend.location = "top_right"
            p2.legend.click_policy = "hide"
            p3.legend.location = "top_right" 
            p3.legend.click_policy = "hide"
            
        else:
            # For smaller datasets, use regular Bokeh plots
            # J-V relationship plot
            p2 = figure(title=f"{title} - Current-Voltage Relationship", 
                        x_axis_label='Voltage (VRHE)', 
                        y_axis_label='Current Density (mA/cm²geo)',
                        width=800, 
                        height=400,
                        tools='pan,wheel_zoom,box_zoom,reset,save')
            
            # Add each waveform to the JV plot
            for i, pulse in enumerate(pulses):
                p2.line(pulse['Voltage (V)'], pulse['Current Density (mA/cm^2)'], 
                      line_width=2, color=colors[i], legend_label=legend[i])
            
            # Configure legend
            p2.legend.location = "top_right"
            p2.legend.click_policy = "hide"
            
            # Charge-Voltage relationship plot
            p3 = figure(title=f"{title} - Charge-Voltage Relationship", 
                        x_axis_label='Voltage (VRHE)', 
                        y_axis_label='Charge Density (mC/cm²geo)',
                        width=800, 
                        height=400,
                        tools='pan,wheel_zoom,box_zoom,reset,save')
            
            # Add each waveform to the charge plot
            for i, pulse in enumerate(pulses):
                p3.line(pulse['Voltage (V)'], pulse['Charge Density (mC/cm^2)'], 
                      line_width=2, color=colors[i], legend_label=legend[i])
            
            # Configure legend
            p3.legend.location = "top_right"
            p3.legend.click_policy = "hide"
        
        # Add reference lines to both plots
        for plot in [p2, p3]:
            # Horizontal zero line
            plot.line([min_voltage, max_voltage], [0, 0], 
                     line_width=1, color='black')
            
            # Vertical zero line
            plot.line([0, 0], [plot.y_range.start, plot.y_range.end], 
                     line_width=1, color='black')
            
            # Add vertical reference lines
            for v in [-1.965, 6.535, -0.335]:
                if min_voltage <= v <= max_voltage:
                    plot.line([v, v], [plot.y_range.start, plot.y_range.end], 
                             line_width=1, color='black', line_dash='dashed')
        
        # Show the plots
        show(column(p2, p3))
        
        return p1, p2, p3
    
    return p1

def getEISFromWaveform(pulse: pd.DataFrame, freqRange: list[float] = None):
    """Experimental, unfinished function to get EIS from waveform by taking fft and doing Z = V/I

    Args:
        pulse (pd.DataFrame): Waveform from readOSC.py
        freqRange (list[float], default None): Constrains output to certain frequencies.
        
    Returns:
        impedance (pd.DataFrame): impedance in same format as output of ReadDataFiles.readPEIS()
    """
    
    t = pulse['Time (s)'].to_numpy()
    i = pulse['Current (A)'].to_numpy()
    v = pulse['Voltage (V)'].to_numpy()
    
    # i = i - np.median(i)
    # v = v - np.median(v)
    dt = np.median(np.diff(t))
    
    #no windowing necessary as this is a repeating function. at different freq's this may not be the case
    
    v_fft = sc.fft.rfft(v)
    i_fft = sc.fft.rfft(i)
    freqs = sc.fft.rfftfreq(len(v),dt)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        z_fft = v_fft / i_fft
        z_fft[~np.isfinite(z_fft)] = np.nan     # NaN out 0/0 and V/0 bins
    
    impedance = pd.DataFrame()
    impedance['freq/Hz'] = freqs
    impedance['Re(Z)/Ohm'] = z_fft.real
    impedance['-Im(Z)/Ohm'] = -z_fft.imag
    
    if freqRange != None:
        impedance = impedance[(impedance['freq/Hz'] >= freqRange[0]) & (impedance['freq/Hz'] <= freqRange[1])]

    
    
    return impedance

def predictWaveform(pulse: pd.DataFrame, eisData: pd.DataFrame):
    """Using EIS data and a voltage waveform, predicts the resulting current waveform.

    Args:
        pulse (pd.DataFrame): Voltage waveform from ReadDataFiles.readRawWaveform()
        eisData (pd.DataFrame): EIS data from ReadDataFiles.readPEIS()
        
    Returns:
        pulse (pd.DataFrame): In the style of ReadDataFiles.readOSC()
    """
    
    return

def getPUND(df: pd.DataFrame, frequency: float = None):
    """

    Args:
        df (pd.DataFrame): Full PUND dataframe from ReadDataFiles.readOSC()
        frequency (float): frequency of waveform in Hz, useful for when it needs to be truncated

    Returns:
        P, N (pd.Dataframe): up and down polarization dataframes. stored in 'Switching Polarization (uC/cm^2)' column
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    #below code truncates dataframe to only have one period
    period_length_in_s = 1/frequency
    df = df[df['Time (s)'] <= period_length_in_s]
    
    # Calculate how many rows we need to add to make it divisible by 4
    n = len(df)
    remainder = n % 4
    
    if remainder != 0:
        rows_to_add = 4 - remainder
        # Method 1: Repeat the last row instead of creating NaN rows
        last_row = df.iloc[[-1]]  # Get last row as DataFrame
        padding = pd.concat([last_row] * rows_to_add, ignore_index=True)
        df = pd.concat([df, padding], ignore_index=True)
    
    
    # Now split into 4 equal parts using np.array_split (will work now)
    # or continue with iloc method
    chunk_size = len(df) // 4
    
    P = df.iloc[0:chunk_size].copy()
    U = df.iloc[chunk_size:2*chunk_size].copy()
    N = df.iloc[2*chunk_size:3*chunk_size].copy()
    D = df.iloc[3*chunk_size:].copy()
    
    #resets time and indices (s is not implemented, not necessary)
    P['Time (ms)'] = P['Time (ms)'] - P['Time (ms)'].iloc[0]
    U['Time (ms)'] = U['Time (ms)'] - U['Time (ms)'].iloc[0]
    N['Time (ms)'] = N['Time (ms)'] - N['Time (ms)'].iloc[0]
    D['Time (ms)'] = D['Time (ms)'] - D['Time (ms)'].iloc[0]

    P = P.reset_index(drop=True)
    U = U.reset_index(drop=True)
    N = N.reset_index(drop=True)
    D = D.reset_index(drop=True)
    
    #calculates switching current density of P and N pulses
    P['Switching Current Density (mA/cm^2)'] = P['Current Density (mA/cm^2)']-U['Current Density (mA/cm^2)']
    N['Switching Current Density (mA/cm^2)'] = N['Current Density (mA/cm^2)']-D['Current Density (mA/cm^2)']
    
    #get rid of this in final code, should be already imported
    
    #integrates switching current to get switching polarization (integrated cumulatively so it can be plotted, but total integral should be the one)
    P['Switching Polarization (uC/cm^2)'] = sc.integrate.cumulative_trapezoid(P['Switching Current Density (mA/cm^2)'],
                                                                 P['Time (ms)'],
                                                                 initial=0)
    N['Switching Polarization (uC/cm^2)'] = sc.integrate.cumulative_trapezoid(N['Switching Current Density (mA/cm^2)'],
                                                                 N['Time (ms)'],
                                                                 initial=0)
    
    return (P, N)

def plotOnePUND(dfs: list[pd.DataFrame], title: str, positiveCurrent: bool = True):
    """Plots one PUND to debug.

    Args:
        dfs (list[pd.DataFrame]): List of P and N dataframes with FE switching polarization columns from getPUND
        title (str): Title of the plot.
        positiveCurrent (bool, optional): Whether current of N is positive or negative. Defaults to True.

    Returns:
        (fig, ax): matplotlib figure and axes
    """
    #unpacks output of getPUND
    P, N = dfs
    
    #plots figure
    fig, ax = plt.subplots()
    ax.plot(P['Time (ms)'],P['Switching Polarization (uC/cm^2)'],color='red',label='P-U')
    
    if positiveCurrent:
        ax.plot(N['Time (ms)'],N['Switching Polarization (uC/cm^2)'],color='k',label='N-D')
    else:
        ax.plot(N['Time (ms)'],-N['Switching Polarization (uC/cm^2)'],color='k',label='-(N-D)')
    ax.set(ylabel=r'FE Switching Polarization $\left(\frac{\mu C}{cm^2_{geo}}\right)$',
           title=title,
           xlabel='Time From Beginning of Pulse (ms)')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plotManyPUNDs(dfss: list[list[pd.DataFrame]],title,positiveCurrent=True,legendList=None,customColors=[]):
    """Plots many PUNDs at the same time.

    Args:
        dfss (listlist[[pd.DataFrame]]): List of list of P and N dataframes with FE switching polarization columns from getPUND formatted into a list.
        title (str): Title of the plot.
        positiveCurrent (bool, optional): Whether current of N is positive or negative. Defaults to True.
        legendList (list[str], optional): What to put in legend, same order as dfss. Defaults to None.
        customColors (list[str], optional): List of custom colors to plot. Defaults to None.

    Returns:
        fig, ax: matplotlib fig and ax
    """
    fig, ax = plt.subplots()
    
    totalIndices = len(dfss)
    
    print('Legend Value, Positive Polarization, Negative Polarization\n')
    
    for i, dfs in enumerate(dfss):
        
        P, N = dfs
        
        if len(customColors) > 0:
            color = customColors[i]
        else:
            color = colorFader('blue','red',i,totalIndices)
            
        if legendList != None:
            legendItem = legendList[i]
        else:
            legendItem= '_'
            
        ax.plot(P['Time (ms)'],P['Switching Polarization (uC/cm^2)'],color=color,label=legendItem)
        if positiveCurrent:
            ax.plot(N['Time (ms)'],N['Switching Polarization (uC/cm^2)'],color=color,label='_')
        else:
            ax.plot(N['Time (ms)'],-N['Switching Polarization (uC/cm^2)'],color=color,label='_')
            
        print(legendItem+', '+str(P['Switching Polarization (uC/cm^2)'].iloc[-1])+', '+str(N['Switching Polarization (uC/cm^2)'].iloc[-1])+'\n')
            
        
    ax.set(ylabel=r'FE Switching Polarization $\left(\frac{\mu C}{cm^2_{geo}}\right)$',
           xlabel='Time From Beginning of Pulse (ms)',
           title=title)
            
    ax.legend()
    
    return fig, ax

def plotOnePUNDTriangle(dfs: list[pd.DataFrame], title: str):
    """Plots one PUND to debug.

    Args:
        dfs (list[pd.DataFrame]): List of P and N dataframes with FE switching polarization columns from getPUND
        title (str): Title of the plot.

    Returns:
        (fig, ax): matplotlib figure and axes
    """
    #unpacks output of getPUND
    P, N = dfs
    
    #uses P and N to plot voltage vs. polarization graph    
    fig, ax = plt.subplots()
    
    ax.plot(P['Voltage (V)'],P['Switching Polarization (uC/cm^2)'],color='red')
    ax.plot(N['Voltage (V)'],N['Switching Polarization (uC/cm^2)'],color='k')
    
    ax.set(ylabel=r'Polarization $\left(\frac{\mu C}{cm^2_{geo}}\right)$',
           title=title,
           xlabel=r'Voltage ($V_{RHE}$)')
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plotManyPUNDTriangles(dfss: list[list[pd.DataFrame]],title,positiveCurrent=True,legendList=None,customColors=[]):
    """Plots many PUNDs at the same time.

    Args:
        dfss (listlist[[pd.DataFrame]]): List of list of P and N dataframes with FE switching polarization columns from getPUND formatted into a list.
        title (str): Title of the plot.
        positiveCurrent (bool, optional): Whether current of N is positive or negative. Defaults to True.
        legendList (list[str], optional): What to put in legend, same order as dfss. Defaults to None.
        customColors (list[str], optional): List of custom colors to plot. Defaults to None.

    Returns:
        fig, ax: matplotlib fig and ax
    """
    fig, ax = plt.subplots()
    
    totalIndices = len(dfss)
    
    print('Legend Value, Positive Polarization, Negative Polarization\n')
    
    
    for i, dfs in enumerate(dfss):
        
        P, N = dfs
        
        if len(customColors) > 0:
            color = customColors[i]
        else:
            color = colorFader('blue','red',i,totalIndices)
            
        if legendList != None:
            legendItem = legendList[i]
        else:
            legendItem= '_'
            
        ax.plot(P['Voltage (V)'],P['Switching Polarization (uC/cm^2)'],color=color,label=legendItem)
        ax.plot(N['Voltage (V)'],N['Switching Polarization (uC/cm^2)'],color=color,label='_')
            
        print(legendItem+', '+str(P['Switching Polarization (uC/cm^2)'].iloc[-1])+', '+str(N['Switching Polarization (uC/cm^2)'].iloc[-1])+'\n')
            
        
    ax.set(ylabel=r'Polarization $\left(\frac{\mu C}{cm^2_{geo}}\right)$',
           title=title,
           xlabel=r'Voltage ($V_{RHE}$)')
    plt.tight_layout()
            
    ax.legend()
    
    plt.show()
    
    return fig, ax 

def decode_encoded_number(encoded):
    """
    Decode a number encoded in the filename format.
    
    Args:
        encoded: String like "1d000000Ep03" or "p0d65" or "n0d5499999999999999"
    
    Returns:
        Float value, or None if parsing fails
    
    Examples:
        "1d000000Ep03" -> 1.000000E+03 -> 1000.0
        "p0d65" -> +0.65 -> 0.65
        "n0d54" -> -0.54 -> -0.54
    """
    # Decode the format:
    # d -> . (decimal point)
    # p -> + (positive)
    # n -> - (negative)
    decoded = encoded.replace('d', '.').replace('p', '+').replace('n', '-')
    
    try:
        return float(decoded)
    except ValueError:
        return None

def extract_frequency(filename):
    """
    Extract the frequency from a filename with format f_XdXXXXEpXX or f_XdXXXXEnXX.
    
    Args:
        filename: String filename (e.g., "33_CA_PUND_f_1d000000Ep03_...")
    
    Returns:
        The frequency as a float, or None if not found
        
    Example:
        "f_1d000000Ep03" -> 1000.0
    """
    match = re.search(r'f_([^_.]+)', filename)
    
    if not match:
        return None
    
    encoded = match.group(1)
    return decode_encoded_number(encoded)

def extract_pulse_width(filename):
    """
    Extract the pulse width from a filename with format PW_XdXXXXEpXX or PW_XdXXXXEnXX.
    
    Args:
        filename: String filename
    
    Returns:
        The pulse width as a float, or None if not found
        
    Example:
        "PW_1d000En04" -> 0.0001
    """
    match = re.search(r'PW_([^_.]+)', filename)
    
    if not match:
        return None
    
    encoded = match.group(1)
    return decode_encoded_number(encoded)

def extract_mid(filename):
    """
    Extract the mid value from a filename with format Mid_pXdXX or Mid_nXdXX.
    
    Args:
        filename: String filename
    
    Returns:
        The mid value as a float, or None if not found
        
    Example:
        "Mid_p0d65" -> 0.65
    """
    match = re.search(r'Mid_([^_.]+)', filename)
    
    if not match:
        return None
    
    encoded = match.group(1)
    return decode_encoded_number(encoded)

def extract_max(filename):
    """
    Extract the max value from a filename with format Max_pXdXX or Max_nXdXX.
    
    Args:
        filename: String filename
    
    Returns:
        The max value as a float, or None if not found
        
    Example:
        "Max_p1d85" -> 1.85
    """
    match = re.search(r'Max_([^_.]+)', filename)
    
    if not match:
        return None
    
    encoded = match.group(1)
    return decode_encoded_number(encoded)

def extract_min(filename):
    """
    Extract the min value from a filename with format Min_pXdXX or Min_nXdXX.
    
    Args:
        filename: String filename
    
    Returns:
        The min value as a float, or None if not found
        
    Example:
        "Min_n0d5499999999999999" -> -0.5499999999999999
    """
    match = re.search(r'Min_([^_.]+)', filename)
    
    if not match:
        return None
    
    encoded = match.group(1)
    return decode_encoded_number(encoded)

def extract_leading_number(filename):
    """
    Extract the leading number from a filename.
    
    Args:
        filename: String filename (e.g., "33_CA_PUND_f_1d000000Ep03.csv")
    
    Returns:
        The leading number as a string, or None if no leading number found
    """
    match = re.match(r'^(\d+)', filename)
    return match.group(1) if match else None

def find_i_range_in_mpt(mpt_filepath):
    """
    Search for the I Range parameter in an .mpt file without loading entire file.
    
    Args:
        mpt_filepath: Path to the .mpt file
    
    Returns:
        The I Range value as a string (e.g., '100 mA', '100 uA', '1 A'), or None if not found
    """
    try:
        with open(mpt_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Look for lines that start with "I Range" (possibly with leading whitespace)
                if line.strip().startswith('I Range') and \
                   not line.strip().startswith('I Range min') and \
                   not line.strip().startswith('I Range max') and \
                   not line.strip().startswith('I Range init'):
                    # Extract the value after "I Range"
                    # Format is typically: "I Range             100 mA"
                    parts = line.split('I Range', 1)
                    if len(parts) > 1:
                        value = parts[1].strip()
                        return value
    except Exception as e:
        print(f"Error reading {mpt_filepath}: {e}")
        return None
    
    return None

def extract_osc_params(csv_filepath, folder_path):
    """
    Get the I Range and all oscillation parameters from the filename and corresponding .mpt file.
    
    Args:
        csv_filepath: Path to the CSV file (e.g., "/path/to/33_file.csv" or just "33_file.csv")
        folder_path: Path to the folder containing the .mpt files
    
    Returns:
        Dictionary with the following keys:
        {
            'i_range': str (e.g., '100mA', '100uA', '1A') or None,
            'frequency': float (e.g., 1000.0) or None,
            'pulse_width': float (e.g., 0.0001) or None,
            'mid': float (e.g., 0.65) or None,
            'max': float (e.g., 1.85) or None,
            'min': float (e.g., -0.55) or None
        }
    """
    # Extract just the filename if a full path was provided
    csv_filename = Path(csv_filepath).name
    
    # Extract the leading number from the CSV filename
    leading_number = extract_leading_number(csv_filename)
    
    if leading_number is None:
        print(f"Warning: No leading number found in '{csv_filename}'")
        return {
            'i_range': None,
            'frequency': None,
            'pulse_width': None,
            'mid': None,
            'max': None,
            'min': None
        }
    
    # Extract all parameters from the CSV filename
    frequency = extract_frequency(csv_filename)
    pulse_width = extract_pulse_width(csv_filename)
    mid = extract_mid(csv_filename)
    max_val = extract_max(csv_filename)
    min_val = extract_min(csv_filename)
    
    # Look for matching .mpt file(s) in the folder
    folder = Path(folder_path)
    pattern = f"{leading_number}_*.mpt"
    mpt_files = list(folder.glob(pattern))
    
    if not mpt_files:
        print(f"Warning: No matching .mpt files found for '{csv_filename}' (looking for {pattern})")
        return {
            'i_range': None,
            'frequency': frequency,
            'pulse_width': pulse_width,
            'mid': mid,
            'max': max_val,
            'min': min_val
        }
    
    # If multiple .mpt files match, use the first one
    mpt_file = mpt_files[0]
    
    if len(mpt_files) > 1:
        print(f"Info: Multiple .mpt files found for '{csv_filename}', using {mpt_file.name}")
    
    # Find the I Range
    i_range = find_i_range_in_mpt(mpt_file)
    
    # Remove spaces from I Range
    if i_range:
        i_range = ''.join(i_range.split())
    
    return {
        'i_range': i_range,
        'frequency': frequency,
        'pulse_width': pulse_width,
        'mid': mid,
        'max': max_val,
        'min': min_val
    }

#if you would like to use CLI to analyze a waveform can do it
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ph', type=float)
    parser.add_argument('--area', type=float)
    parser.add_argument('--refpot', type=float)
    parser.add_argument('--irange', type=str)
    parser.add_argument('--time', type=float)
    parser.add_argument('--freq', type=float)
    parser.add_argument('--filename', type=str)
    parser.add_argument('--stretch', default=1, type=int)
    args = parser.parse_args()
    
    try:
    
        data = readOSC(args.filename,
                    args.ph,
                    args.area,
                    args.refpot,
                    args.irange,
                    stretch=args.stretch)
        analyzeWaveform(data,
                        args.time,
                        args.freq,
                        args.filename)
        
    except TypeError:
        
        whycantijustterminateapythonscriptmantherehastobeaworkaroundforthisright = 1