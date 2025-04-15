import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
import matplotlib as mpl
from ReadDataFiles import readOSC, readRawWaveform, colorFader, calculateIntegral
import argparse
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column, row
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker, ColumnDataSource, Range1d, HoverTool, Label, LinearAxis
from bokeh.io import output_notebook, reset_output
from bokeh.palettes import Viridis256, Spectral11
import datashader as ds
import datashader.transfer_functions as tf
InteractiveImage = None
from colorcet import fire
import holoviews as hv
from holoviews.operation.datashader import datashade, shade

def fft(data: pd.DataFrame,dataLabel: str,timeLabel: str):
    
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

def plotFFT(datasets: list[tuple],legend,title):
    
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
    plt.show()
    
    return

def analyzeWaveform(pulse: pd.DataFrame, experimentLength: float, frequency: float, title: str = None,plot=True):
    
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

# def plotWaveform(pulse: pd.DataFrame, title: str, jv: bool, reference = pd.DataFrame()):
    
#     fig, ax = plt.subplots()
#     ax2 = ax.twinx()
    
#     #ax.plot(pulse['Time (s)']*1000,pulse['Current Density (mA/cm^2)'],color='r')
#     ax.plot(pulse['Time (s)']*1000,pulse['Current (A)'],color='r')
#     ax2.plot(pulse['Time (s)']*1000,pulse['Voltage (V)'],color='k')
#     if not reference.empty:
#         ax2.plot(reference['Time (s)']*1000,reference['Voltage (V)'],color='k',linestyle='--')
#     ax.set(title= title,
#            #ylabel=r'Current Density $(\frac{mA}{cm^2_{geo}})$',
#            ylabel=r'Current (A)',
#            xlabel='Time (ms)')
#     ax2.set(ylabel = r'Voltage (V$_{RHE}$)')
#     ax.axhline(0,color='k',zorder=0)
#     plt.show()
    
#     if jv == True:
#         fig, ax = plt.subplots()
#         plt.axhline(0,color='k',zorder=0)
#         plt.axvline(0,color='k',zorder=0)
#         plt.axvline(-1.965,color='k',linestyle='--',zorder=0)
#         plt.axvline(6.535,color='k',linestyle='--',zorder=0)
#         plt.axvline(-0.335,color='k',linestyle='--',zorder=0)
#         #plt.scatter(pulse['Voltage (V)'],pulse['Charge Density (mC/cm^2)'],c=(pulse['Time (s)']-pulse['Time (s)'].iloc[0])*1e6,cmap='gist_rainbow')
#         plt.scatter(pulse['Voltage (V)'],pulse['Current Density (mA/cm^2)'],c=(pulse['Time (s)']-pulse['Time (s)'].iloc[0])*1e6,cmap='gist_rainbow')
#         plt.colorbar(label=r'Time ($\mu$s)')
#         plt.title(title)
#         plt.xlabel(r'Voltage (V$_{RHE}$)')
#         #plt.ylabel(r'Charge Density $(\frac{mC}{cm^2_{geo}})$')
#         plt.ylabel(r'Current Density $(\frac{mA}{cm^2_{geo}})$')
#         plt.show()
    
#     return

# def plotWaveforms(pulses: list[pd.DataFrame], title: str, legend: list[str], jv: bool, reference = pd.DataFrame(), customColors = None):
    
#     fig, ax = plt.subplots()
#     ax2 = ax.twinx()
    
    
    
#     for i, pulse in enumerate(pulses):
#         if customColors == None:
#             color = colorFader('blue','red',i,len(pulses))
#         else:
#             color = customColors[i]
#         ax.plot(pulse['Time (s)']*1000,pulse['Current Density (mA/cm^2)'],color=color)
#         ax2.plot(pulse['Time (s)']*1000,pulse['Voltage (V)'],color=color,linestyle=':')
        
#     if not reference.empty:
#         ax2.plot(reference['Time (s)']*1000,reference['Voltage (V)'],color='k',linestyle='--')
    
#     ax.set(title= title,
#         ylabel=r'Current Density $(\frac{mA}{cm^2_{geo}})$',
#         xlabel='Time (ms)')
#     ax.legend(legend)
#     ax2.set(ylabel = r'Voltage (V$_{RHE}$)')
#     ax.axhline(0,color='k',zorder=0)
#     plt.show()
    
#     if jv == True:
#         fig, ax = plt.subplots()
        
#         for i, pulse in enumerate(pulses):
#             if customColors == None:
#                 color = colorFader('blue','red',i,len(pulses))
#             else:
#                 color = customColors[i]
#             ax.plot(pulse['Voltage (V)'],pulse['Current Density (mA/cm^2)'],color=color)
#         ax.legend(legend)
#         ax.set(title = title,
#                ylabel = r'Current Density $(\frac{mA}{cm^2_{geo}})$',
#                xlabel = 'Voltage (V)')
        
#         ax.axhline(0,color='k',zorder=0)
#         ax.axvline(0,color='k',zorder=0)
#         ax.axvline(-1.965,color='k',linestyle='--',zorder=0)
#         ax.axvline(6.535,color='k',linestyle='--',zorder=0)
#         ax.axvline(-0.335,color='k',linestyle='--',zorder=0)
#         plt.show()
        
#         fig, ax = plt.subplots()
        
#         for i, pulse in enumerate(pulses):
#             if customColors == None:
#                 color = colorFader('blue','red',i,len(pulses))
#             else:
#                 color = customColors[i]
#             ax.plot(pulse['Voltage (V)'],pulse['Charge Density (mC/cm^2)'],color=color)
#         ax.legend(legend)
#         ax.set(title = title,
#                ylabel = r'Charge Density $(\frac{mC}{cm^2_{geo}})$',
#                xlabel = 'Voltage (V)')
#         ax.axhline(0,color='k',zorder=0)
#         ax.axvline(0,color='k',zorder=0)
#         ax.axvline(-1.965,color='k',linestyle='--',zorder=0)
#         ax.axvline(6.535,color='k',linestyle='--',zorder=0)
#         ax.axvline(-0.335,color='k',linestyle='--',zorder=0)
#         plt.show()
    
#     return

def plotWaveform(pulse: pd.DataFrame, title: str, jv: bool, reference=pd.DataFrame()):
    """
    Plot waveform data using Bokeh with improved formatting
    
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
    Plot multiple waveforms using Bokeh and Datashader for performance
    
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
        if len(pulses) <= 11:
            colors = Spectral11[:len(pulses)]
        else:
            # Generate colors using colorFader for more than 11 datasets
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
                width=800, height=400,
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
                width=800, height=400,
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

def getEISFromWaveform(pulse: pd.DataFrame):
    
    return

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