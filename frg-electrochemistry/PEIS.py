from ReadDataFiles import readPEIS, colorFader, readPEISPandas, readRawWaveform
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from impedance.models.circuits import CustomCircuit
from impedance.preprocessing import cropFrequencies, ignoreBelowX
import os
from pyDRTtools.runs import EIS_object, simple_run
from scipy.fft import irfft, rfftfreq, rfft, ifft
from OSC import plotWaveform

def plotOneBode(data,title):
    """Takes dataframe from readPEISPandas. May rewrite in the future to accept f, Z values.

    Args:
        data (pd.DataFrame): From readPEISPandas.
        title (str): title + ' Bode Plot'
    """
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_xscale('log')
    ax.plot(data['freq/Hz'],data['|Z|/Ohm'],'k')
    ax.set_yscale('log')
    ax2.plot(data['freq/Hz'],-data['Phase(Z)/deg'],'r')
    ax.set(title = title + ' Bode Plot',
           xlabel = 'Frequency (Hz)',
           ylabel = 'Magnitude ($\Omega$)')
    ax2.set(ylabel = '-Phase (deg)')
    
    plt.show()
    
    return

def generateCircuitFit(f,Z):
    """Takes in data from convertToImpedanceAnalysis and fits it to a specific circuit.

    Args:
        f (np.ndarray[float]): frequencies
        Z (np.ndarray[complex]): impedances
        
    Returns:
        circuit (CustomCircuit): can use this to plot
    """
    #generates circuit model
    circuit = 'p(p(R1,C1),p(R2,CPE2))-R0'
    initialGuess = [400,50e-6,0,0,1,6]
    circuit = CustomCircuit(circuit,initial_guess=initialGuess)
    circuit = circuit.fit(f,Z)
    ZValues1 = circuit.predict(f)
    
    fig, ax = plt.subplots()
    ax.plot(Z.real,-Z.imag,'ko')
    ax.plot(ZValues1.real,-ZValues1.imag,'r-')
    
    plt.show()
    
    return circuit

def plotOneNyquist(filename,title,freqRange,fitModel=False,circuitString = None,initialGuess=[],bounds=([],[])):
    
    fig, ax = plt.subplots()
    f, Z = readPEIS(filename)
    f, Z = cropFrequencies(f,Z,freqRange[0],freqRange[1])
    if fitModel:
        circuit = CustomCircuit(circuitString,initial_guess=initialGuess)
        circuit = circuit.fit(f,Z,bounds)
        zPredict = circuit.predict(f)
        ax.plot(zPredict.real,-zPredict.imag,color='k',linestyle='-',label=circuitString)
    
    ax.plot(Z.real,-Z.imag,'o',color='k')
    ax.set(title = title + ' Nyquist Plots',
           xlabel = 'Re(Z(f)) ($\Omega$)',
           ylabel = '-Im(Z(f)) ($\Omega$)')
    print(circuit)
    plt.show()
    
    return circuit

def plotCompareNyquist(filenames,title,freqRange,fitModel=False,circuitString=None,initialGuess=[],bounds=([],[]),legendList=None):
    
    numberOfPlots = len(filenames)
    circuitList = []
    fig, ax = plt.subplots()
    
    for i in range(0,numberOfPlots):
        
        #gets f and Z values
        f, Z = readPEIS(filenames[i])
        
        #crops frequencies
        f, Z = cropFrequencies(f,Z,freqRange[0],freqRange[1])
        
        if fitModel:
            #generates circuit model
            #could implement smarter way of getting initial guesses from data
            circuit = CustomCircuit(circuitString,initial_guess=initialGuess)
            
            #fits value to circuit
            circuit = circuit.fit(f,Z,bounds)
            circuitList.append(circuit)
            
            #gets circuit predicted Z values
            zPredict = circuit.predict(f)
        
        #plots results
        color = colorFader('blue','red',i/(numberOfPlots-1))
        if legendList != None:
            ax.plot(Z.real,-Z.imag,'o',color=color,label=legendList[i])
        else:
            ax.plot(Z.real,-Z.imag,'o',color=color)
        if fitModel:
            ax.plot(zPredict.real,-zPredict.imag,color=color,linestyle='-',label='_')
        
    
    maxBounds = max([ax.get_ylim()[1],ax.get_xlim()[1]])
    
    ax.set(title = title + ' Nyquist Plots',
           xlabel = 'Re(Z(f)) ($\Omega$)',
           ylabel = '-Im(Z(f)) ($\Omega$)')
    
    if legendList != None:
        ax.legend()
    
    plt.show()
    
    return circuitList

def plotCircuitProperties(circuitList):
    
    numberOfCircuits = len(circuitList)
    
    names = circuitList[0].get_param_names()[0]
    units = circuitList[0].get_param_names()[1]
    
    numberOfParameters = len(names)
    
    parameterMatrix = np.zeros((numberOfParameters,numberOfCircuits))
    confMatrix = np.zeros((numberOfParameters,numberOfCircuits))
    
    for i in range(0,numberOfParameters):
        for j in range(0,numberOfCircuits):
            parameterMatrix[i,j] = circuitList[j].parameters_[i]
            confMatrix[i,j] = circuitList[j].conf_[i]
    
    for i, parameter in enumerate(names):
        
        fig, ax = plt.subplots()
        ax.errorbar(range(numberOfCircuits),
                    parameterMatrix[i],
                    yerr=confMatrix[i],
                    color='k',
                    fmt='o-',
                    capsize=5)
        ax.set(title=parameter,
               ylabel = parameter + ' (' + units[i] + ')',
               xlabel = 'Circuit Index')
        plt.show()
    
    return

def convertToPyDRTTools(filename:str,freqRange: list[float]):
    
    data = readPEISPandas(filename)
    data['Im(Z)/Ohm'] = -data['-Im(Z)/Ohm']   
    data = data[(data['freq/Hz'] > freqRange[0]) & (data['freq/Hz'] < freqRange[1])] 
    data.to_csv(filename[:-4]+'.csv',
                sep=',',
                index=False,
                header=False,
                columns=['freq/Hz','Re(Z)/Ohm','Im(Z)/Ohm'])
    
    return

def processFolder(foldername: str, freqRange):
    #written by chatGPT
    peis_files = []

    # Walk through the folder and its subdirectories
    for root, dirs, files in os.walk(foldername):
        for file in files:
            # Check if the file name contains 'PEIS' and ends with '.txt'
            if 'PEIS' in file and file.endswith('.txt'):
                filepath = os.path.join(root, file)
                peis_files.append(filepath)

    for file in peis_files:
        convertToPyDRTTools(file,freqRange)
    
    return

def plotCompareDRT(filenames,title,freqRange,legendList=None):
    
    numberOfPlots = len(filenames)
    fig, ax = plt.subplots()
    ax.set_title(title)
    
    for i in range(0,numberOfPlots):
        
        data = readPEISPandas(filenames[i])
        data['Im(Z)/Ohm'] = -data['-Im(Z)/Ohm']   
        data = data[(data['freq/Hz'] > freqRange[0]) & (data['freq/Hz'] < freqRange[1])] 
        
        data = EIS_object(data['freq/Hz'].to_numpy(),
                          data['Re(Z)/Ohm'].to_numpy(),
                          data['Im(Z)/Ohm'].to_numpy())
        data = simple_run(data,
                          rbf_type='Gaussian',
                          data_used='Combined Re-Im Data',
                          induct_used=0,
                          der_used='1st order',
                          cv_type='LC',
                          reg_param= 1E-4,
                          shape_control='FWHM Coefficient',
                          coeff=0.1)
        
        #data = readDRT(filenames[i])
        color = colorFader('blue','red',i/(numberOfPlots-1))
        freq = 1 / data.out_tau_vec
        ax.plot(freq, data.gamma,color=color)
    
    ax.set(ylabel = r'$\gamma$ ($\Omega$)',
           xlabel = r'Frequency (Hz)',
           xscale='log')
    if legendList != None:
        ax.set_legend(legendList)
        
    plt.show()
    
    return

def predictCurrent(peisFilename,rawWaveformFilename):
    
    #use only seconds and hertz for units to work out
    peisData = readPEISPandas(peisFilename)
    peisData = peisData.sort_values('freq/Hz')
    voltageWaveform = readRawWaveform(rawWaveformFilename,14,0.197)
    dataLength = voltageWaveform.shape[0]
    
    #finds dt, distance between time points
    dt = voltageWaveform['Time (s)'].diff().mean()
    
    voltageFFT = rfft(voltageWaveform['RawVoltage (V)'].to_numpy())
    voltageFFTFreq = rfftfreq(dataLength, dt)
    print(len(voltageFFT),len(voltageFFTFreq))
    currentFFT = np.zeros(voltageFFT.size,dtype=complex)
    
    print(peisData['freq/Hz'].min(),voltageFFTFreq.min())
    print(peisData['freq/Hz'].max(),voltageFFTFreq.max())
    
    for i in range(len(voltageFFT)):
        #need to account for negative frequencies - how to mirror peisData to deal with these?
        #linearly interpolates peisData to find best value based on frequency
        imagImpedance = -np.interp(voltageFFTFreq[i],peisData['freq/Hz'],peisData['-Im(Z)/Ohm'])
        realImpedance = np.interp(voltageFFTFreq[i],peisData['freq/Hz'],peisData['Re(Z)/Ohm'])
        impedancePhasor = complex(realImpedance,imagImpedance)
        voltagePhasor = voltageFFT[i]
        
        currentFFT[i] = voltagePhasor/impedancePhasor #V/Z = I
        
        if i%1000 == 0:
            print(voltageFFTFreq[i])
            print(voltagePhasor,impedancePhasor)
            print(currentFFT[i])
        
    currentSignal = irfft(currentFFT,n=dataLength)
    
    dataframe = pd.DataFrame({'Time (s)':voltageWaveform['Time (s)'],
                              'Current Density (mA/cm^2)':currentSignal,
                              'Voltage (V)':voltageWaveform['RawVoltage (V)']})
    plotWaveform(dataframe,'title',False)
        
    return

# circuitList = plotCompareNyquist([r'Data_Files\2024-07-31-TN-01-050\6_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Data_Files\2024-07-31-TN-01-050\18_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Data_Files\2024-08-01-TN-01-051\5_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Data_Files\2024-08-01-TN-01-051\15_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Data_Files\2024-08-04-TN-01-052\6_PEIS_HER_After_Debubbling_02_PEIS_C01.txt',
#                                   r'Data_Files\2024-08-04-TN-01-052\15_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Data_Files\2024-08-05-TN-01-053\6_PEIS_HER_afterdebubbling_02_PEIS_C01.txt',
#                                   r'Data_Files\2024-08-05-TN-01-053\16_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Data_Files\2024-09-04-TN-01-054\5_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Data_Files\2024-09-04-TN-01-054\15_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Data_Files\2024-09-09-TN-01-055\5_PEIS_HER_C01.txt',
#                                   r'Data_Files\2024-09-09-TN-01-055\13_PEIS_HER_C01.txt'
#                                   ],
#                                 'Degradation of N&S Sample',
#                                 [3,100000],
#                                 fitModel=True,
#                                 circuitString='p(R1,CPE1)-R0',
#                                 bounds = ([0,0,0,5],[1000,150e-6,1,15]),
#                                 initialGuess=[250,50e-6,1,9])
# plotCircuitProperties(circuitList)

# circuitList = plotCompareNyquist([r'Data_Files\2024-06-18-TN-01-044\4_PEIS_HER_C02.txt',
#                                   r'Data_Files\2024-06-18-TN-01-044\14_PEIS_HER_02_PEIS_C02.txt',
#                                   #r'Joana Data\2024-07-09-JD-01-008\7_PEIS_HER_02_PEIS_C01.txt',
#                                   #r'Joana Data\2024-07-09-JD-01-008\17_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Joana Data\2024-07-12-JD-01-009\5_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Joana Data\2024-07-12-JD-01-009\15_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Joana Data\2024-07-18-JD-01-010\5_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Joana Data\2024-07-18-JD-01-010\10_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Joana Data\2024-07-23-JD-01-010\5_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Joana Data\2024-07-24-JD-01-011\7_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Joana Data\2024-07-24-JD-01-011\13_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Joana Data\2024-07-29-JD-01-011-mrb230410Cii\5_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Joana Data\2024-07-30-JD-01-011\5_PEIS_HER_02_PEIS_C01.txt',
#                                   r'Joana Data\2024-07-30-JD-01-011\14_PEIS_HER_02_PEIS_C01.txt',
#                                  ],
#                                 'Degradation of Matt\'s 2nd Sample',
#                                 [2,200000],
#                                 fitModel=True,
#                                 circuitString='p(R1,CPE1)-R0',
#                                 bounds = ([500,0,0,12],[8000,1000e-6,1,70]),
#                                 initialGuess=[1000,50e-6,1,16])
# plotCircuitProperties(circuitList)

#processFolder(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files',[3,7000000])

# plotCompareDRT([#r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-07-31-TN-01-050\6_PEIS_HER_02_PEIS_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-07-31-TN-01-050\18_PEIS_HER_02_PEIS_C01.txt',
#                 #r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-01-TN-01-051\5_PEIS_HER_02_PEIS_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-01-TN-01-051\15_PEIS_HER_02_PEIS_C01.txt',
#                 #r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\6_PEIS_HER_After_Debubbling_02_PEIS_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\15_PEIS_HER_02_PEIS_C01.txt',
#                 #r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-05-TN-01-053\6_PEIS_HER_afterdebubbling_02_PEIS_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-05-TN-01-053\16_PEIS_HER_02_PEIS_C01.txt',
#                 #r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-09-04-TN-01-054\5_PEIS_HER_02_PEIS_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-09-04-TN-01-054\15_PEIS_HER_02_PEIS_C01.txt',
#                 #r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-09-09-TN-01-055\5_PEIS_HER_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-09-09-TN-01-055\13_PEIS_HER_C01.txt'
#                 ],
#                'DRT Evolution of N&S',
#                [3,200000])

predictCurrent(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-07-31-TN-01-050\18_PEIS_HER_02_PEIS_C01.txt',
               r'C:\Users\tejas\Analysis\Potentiostat\Waveforms\f_1d000000Ep03_PW_1d000En04_F_n1d37_U_p5d5_D_n3d0.csv')