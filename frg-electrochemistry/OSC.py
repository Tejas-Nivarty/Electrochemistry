import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
import matplotlib as mpl
from ReadDataFiles import readOSC, readRawWaveform, colorFader, calculateIntegral
import argparse

def fft(data: pd.DataFrame,dataLabel: str):
    
    dt = data['Time (s)'].diff().mean()
    
    inputArray = np.array(data[dataLabel])
    inputArray = np.tile(inputArray, 20)
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
    
    for i, data in enumerate(datasets):
        
        color = colorFader('blue','red',i,len(datasets))
        frequencies = data[0]
        magnitude = data[1]
        if maxMagnitude < max(magnitude):
            maxMagnitude = max(magnitude)
        ax.scatter(frequencies,magnitude,
                   color=color,
                   s=0.1,
                   label=legend[i])
    
    ax.set(title = title,
           xlabel = 'Frequency (Hz)',
           ylabel = 'Magnitude',
           xscale = 'log',
           xlim = [10**3,125000000/2],
           yscale = 'log',
           ylim = [10**-6,1])
    ax.legend()
    plt.show()
    
    return

def analyzeWaveform(pulse: pd.DataFrame, experimentLength: float, frequency: float, title: str):
    
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
    
    plotWaveform(pulse,title,jv=True)

    return

def plotWaveform(pulse: pd.DataFrame, title: str, jv: bool, reference = pd.DataFrame()):
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    
    #ax.plot(pulse['Time (s)']*1000,pulse['Current Density (mA/cm^2)'],color='r')
    ax.plot(pulse['Time (s)']*1000,pulse['Current (A)'],color='r')
    ax2.plot(pulse['Time (s)']*1000,pulse['Voltage (V)'],color='k')
    if not reference.empty:
        ax2.plot(reference['Time (s)']*1000,reference['Voltage (V)'],color='k',linestyle='--')
    ax.set(title= title,
           #ylabel=r'Current Density $(\frac{mA}{cm^2_{geo}})$',
           ylabel=r'Current (A)',
           xlabel='Time (ms)')
    ax2.set(ylabel = r'Voltage (V$_{RHE}$)')
    ax.axhline(0,color='k',zorder=0)
    plt.show()
    
    if jv == True:
        fig, ax = plt.subplots()
        plt.axhline(0,color='k',zorder=0)
        plt.axvline(0,color='k',zorder=0)
        plt.axvline(-1.965,color='k',linestyle='--',zorder=0)
        plt.axvline(6.535,color='k',linestyle='--',zorder=0)
        plt.axvline(-0.335,color='k',linestyle='--',zorder=0)
        #plt.scatter(pulse['Voltage (V)'],pulse['Charge Density (mC/cm^2)'],c=(pulse['Time (s)']-pulse['Time (s)'].iloc[0])*1e6,cmap='gist_rainbow')
        plt.scatter(pulse['Voltage (V)'],pulse['Current Density (mA/cm^2)'],c=(pulse['Time (s)']-pulse['Time (s)'].iloc[0])*1e6,cmap='gist_rainbow')
        plt.colorbar(label=r'Time ($\mu$s)')
        plt.title(title)
        plt.xlabel(r'Voltage (V$_{RHE}$)')
        #plt.ylabel(r'Charge Density $(\frac{mC}{cm^2_{geo}})$')
        plt.ylabel(r'Current Density $(\frac{mA}{cm^2_{geo}})$')
        plt.show()
    
    return

def plotWaveforms(pulses: list[pd.DataFrame], title: str, legend: list[str], jv: bool, reference = pd.DataFrame(), customColors = None):
    
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
        plt.show()
    
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