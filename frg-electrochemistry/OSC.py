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

booster1us = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-11-13-TN-01-060\8_Dynamic_CA_1000Hz_1us.csv',
                     14,
                     0.17343772,
                     0.217,
                     '2 A')
booster10us = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-11-13-TN-01-060\9_Dynamic_CA_1000Hz_10us.csv',
                     14,
                     0.17343772,
                     0.217,
                     '2 A')
booster100us = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-11-13-TN-01-060\10_Dynamic_CA_1000Hz_100us.csv',
                     14,
                     0.17343772,
                     0.217,
                     '2 A')
# nobooster1us = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-05-TN-01-053\8_1000Hz_PW-6_Dynamic_CA.csv',
#                        14,
#                        0.1826403875,
#                        0.209,
#                        '1 A')
# nobooster10us = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-05-TN-01-053\7_1000Hz_PW-5_Dynamic_CA.csv',
#                        14,
#                        0.1826403875,
#                        0.209,
#                        '1 A')
nobooster100us = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-07-31-TN-01-050\11_1000Hz_Dynamic_CA.csv',
                       14,
                       0.1826403875,
                       0.209,
                       '1 A')
# booster20us = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-11-13-TN-01-060\11_Dynamic_CA_1000Hz_20us.csv',
#                       14,
#                       0.17343772,
#                       0.217,
#                       '2 A')
booster20usBetterCable = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-11-13-TN-01-060\12_Dynamic_CA_1000Hz_20us_newboostercable.csv',
                                14,
                                0.17343772,
                                0.217,
                                '2 A')
# reference20us = readRawWaveform(r'C:\Users\tejas\Analysis\Potentiostat\Waveforms\f_1d000000Ep03_PW_2d000En05_F_n1d378_U_p5d492_D_n3d008_False.csv',
#                                 14,
#                                 0.217)
reference100us = readRawWaveform(r'C:\Users\tejas\Analysis\Potentiostat\Waveforms\f_1d000000Ep03_PW_1d000En04_F_n1d378_U_p5d492_D_n3d008_False.csv',
                                 14,
                                 0.217)
# reference1us = readRawWaveform(r'C:\Users\tejas\Analysis\Potentiostat\Waveforms\f_1d000000Ep03_PW_1d000En06_F_n1d378_U_p5d492_D_n3d008_False.csv',
#                                14,
#                                0.217)
# reference10us = readRawWaveform(r'C:\Users\tejas\Analysis\Potentiostat\Waveforms\f_1d000000Ep03_PW_1d000En05_F_n1d378_U_p5d492_D_n3d008_False.csv',
#                                 14,
#                                 0.217)
# reference100usNF = readRawWaveform(r'C:\Users\tejas\Analysis\Potentiostat\Waveforms\f_1d000000Ep03_PW_1d000En04_F_n0d8_U_p5d5_D_n3d0_False.csv',
#                                    14,
#                                    0.209)
# reference100usNFOnlyDown = readRawWaveform(r'C:\Users\tejas\Analysis\Potentiostat\Waveforms\f_1d000000Ep03_PW_1d000En04_F_n0d8_U_n0d8_D_n3d0_False.csv',
#                                            14,
#                                            0.209)
# NF100us = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-09-04-TN-01-054\9_1000Hz_1E-4_NonFaradaic.csv',
#                   14,
#                   0.1826403875,
#                   0.209,
#                   '1 A')
# NFOnlyDown100us = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-09-04-TN-01-054\8_1000Hz_1E-4_OnlyDown.csv',
#                           14,
#                           0.1826403875,
#                           0.209,
#                           '1 A')

# plotWaveform(NF100us,
#              '1 kHz, 100 $\mu$s Non-Faradaic Up and Down',
#              True,
#              reference=reference100usNF)
# plotWaveform(NFOnlyDown100us,
#              '1 kHz, 100 $\mu$s Non-Faradaic Only Down Pulse',
#              True,
#              reference=reference100usNFOnlyDown)
# plotWaveforms([NF100us,
#                NFOnlyDown100us],
#               'Non-Faradaic Comparison',
#               ['Up and Down',
#                'Only Down'],
#               True)

# plotWaveforms([booster1us,
#                booster10us,
#                booster20usBetterCable,
#                booster100us],
#               'Booster Comparison 1 kHz',
#               ['1 $\mu$s',
#                '10 $\mu$s',
#                '20 $\mu$s',
#                '100 $\mu$s'],
#               True)

# plotWaveforms([booster1us,
#                nobooster1us],
#               '1 us Comparison',
#               ['Booster','No Booster'],
#               True,
#               reference=reference1us)

# plotWaveforms([booster10us,
#                nobooster10us],
#               'Booster Comparison 10 us',
#               ['Booster',
#                'No Booster'],
#               True,
#               reference=reference10us)

# plotWaveforms([booster20us,
#                booster20usBetterCable],
#               '20 us Pulsing Differences',
#               ['Bad Cable',
#                'Official Cable'],
#               True,
#               reference=reference20us)

# plotWaveforms([booster100us,
#                nobooster100us],
#               '100 us Comparison',
#               ['Booster','No Booster'],
#               True,
#               reference=reference100us)
# plotWaveform(nobooster100us,
#              '1000 Hz, 100 $\mu$s Voltage and Current Response',
#              True,
#              reference=reference100us)

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