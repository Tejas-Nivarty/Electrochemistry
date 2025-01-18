from ReadDataFiles import readPEIS, colorFader, readPEISPandas, readRawWaveform, readOSC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from impedance.models.circuits import CustomCircuit
from impedance.preprocessing import cropFrequencies, ignoreBelowX
import os
from pyDRTtools.runs import EIS_object, simple_run
from scipy.fft import irfft, rfftfreq, rfft, ifft
import scipy as sc
from OSC import plotWaveforms

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
    if fitModel == False:
        circuit = None
    print(circuit)
    plt.show()
    
    return circuit

def plotCompareNyquist(filenames,title,freqRange,fitModel=False,circuitString=None,initialGuess=[],bounds=([],[]),legendList=None,saveData=False):
    
    numberOfPlots = len(filenames)
    circuitList = []
    fig, ax = plt.subplots()
    
    if saveData:
        datadf = pd.DataFrame()
    
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
        color = colorFader('blue','red',i,numberOfPlots)
        if legendList != None:
            ax.plot(Z.real,-Z.imag,'o',color=color,label=legendList[i])
        else:
            ax.plot(Z.real,-Z.imag,'o',color=color)
        if fitModel:
            ax.plot(zPredict.real,-zPredict.imag,color=color,linestyle='-',label='_')
            
        if saveData:
            datadf[legendList[i]+' RealData'] = Z.real
            datadf[legendList[i]+' -ImagData'] = -Z.imag
            if fitModel:
                datadf[legendList[i]+' RealCircuitFit'] = zPredict.real
                datadf[legendList[i]+' -ImagCircuitFit'] = -zPredict.imag 
        
    
    maxBounds = max([ax.get_ylim()[1],ax.get_xlim()[1]])
    
    ax.set(title = title + ' Nyquist Plots',
           xlabel = 'Re(Z(f)) ($\Omega$)',
           ylabel = '-Im(Z(f)) ($\Omega$)')
    
    plt.axis('square')
    
    if legendList != None:
        ax.legend()
    
    plt.show()
    
    if saveData:
        return datadf
    else:
        return circuitList

def plotCircuitProperties(circuitList,legendList,saveData=False):
    
    numberOfCircuits = len(circuitList)
    
    names = circuitList[0].get_param_names()[0]
    units = circuitList[0].get_param_names()[1]
    
    numberOfParameters = len(names)
    
    parameterMatrix = np.zeros((numberOfParameters,numberOfCircuits))
    confMatrix = np.zeros((numberOfParameters,numberOfCircuits))
    
    datadf = None
    
    if saveData:
        datadf = pd.DataFrame()
    
    for i in range(0,numberOfParameters):
        for j in range(0,numberOfCircuits):
            parameterMatrix[i,j] = circuitList[j].parameters_[i]
            confMatrix[i,j] = circuitList[j].conf_[i]
            
    if saveData:
        
        elementStringList = []
        for i, name in enumerate(names):
            elementStringList.append(name + ' ' + units[i])
        
        datadf['Circuit Element'] = elementStringList
        
        for i in range(0,numberOfCircuits):
            
            datadf[legendList[i]+' Value'] = circuitList[i].parameters_
            datadf[legendList[i]+' Err'] = circuitList[i].conf_
                
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
               xticks = range(numberOfCircuits),
               xticklabels = legendList)
        
        plt.show()
    
    return datadf

def convertToPyDRTTools(filename:str,freqRange: list[float]):
    
    data = readPEISPandas(filename)
    data['Im(Z)/Ohm'] = -data['-Im(Z)/Ohm']   
    data = data[(data['freq/Hz'] >= freqRange[0]) & (data['freq/Hz'] <= freqRange[1])] 
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

def plotCompareDRT(filenames,title,freqRange,legendList=None,rbf_type='Gaussian',der_used='1st order',cv_type='GCV',reg_param=1E-4,shape_control='FWHM Coefficient',coeff=0.1):
    
    numberOfPlots = len(filenames)
    fig, ax = plt.subplots()
    ax.set_title(title)
    
    for i in range(0,numberOfPlots):
        
        data = readPEISPandas(filenames[i])
        data['Im(Z)/Ohm'] = -data['-Im(Z)/Ohm']   
        data = data[(data['freq/Hz'] >= freqRange[0]) & (data['freq/Hz'] <= freqRange[1])] 
        
        data = EIS_object(data['freq/Hz'].to_numpy(),
                        data['Re(Z)/Ohm'].to_numpy(),
                        data['Im(Z)/Ohm'].to_numpy())
        data = simple_run(data,
                        rbf_type=rbf_type,
                        data_used='Combined Re-Im Data',
                        induct_used=0,
                        der_used=der_used,
                        cv_type=cv_type,
                        reg_param= reg_param, #1E-4 is good
                        shape_control=shape_control,
                        coeff=coeff) #0.3 is good
        color = colorFader('blue','red',i,numberOfPlots)
        freq = 1 / data.out_tau_vec
        ax.plot(data.out_tau_vec, data.gamma,color=color)
    
    ax.set(ylabel = r'$\gamma$ ($\Omega$)',
           xlabel = r'RC Time Constant (s)',
           xscale='log')
    if legendList != None:
        ax.legend(legendList)
        
    plt.show()
    
    return

def predictCurrent(peisFilename,voltageWaveformFilename,pH,area,referencePotential):
    
    #sets DC offset so 0 current is HER equilibrium
    DCOffset = referencePotential + 0.059*pH
    
    #use only seconds and hertz for units to work out
    peisData = readPEISPandas(peisFilename)
    peisData = peisData.sort_values('freq/Hz')
    if 'f_' in voltageWaveformFilename:
        voltageWaveform = readRawWaveform(voltageWaveformFilename,pH,referencePotential)
    else:
        voltageWaveform = readOSC(voltageWaveformFilename,pH,area,referencePotential,'1 A')
    voltageWaveform['RawVoltage (V)'] = voltageWaveform['RawVoltage (V)'] + DCOffset
    dataLength = voltageWaveform.shape[0]
    
    dt = voltageWaveform['Time (s)'].diff().mean()
    
    voltageFFT = rfft(voltageWaveform['RawVoltage (V)'].to_numpy())
    voltageFFTFreq = rfftfreq(dataLength, dt)
    currentFFT = np.zeros(voltageFFT.size,dtype=complex)
    
    for i in range(len(voltageFFT)):
        #linearly interpolates peisData to find best value based on frequency
        imagImpedance = -np.interp(voltageFFTFreq[i],peisData['freq/Hz'],peisData['-Im(Z)/Ohm'])
        realImpedance = np.interp(voltageFFTFreq[i],peisData['freq/Hz'],peisData['Re(Z)/Ohm'])
        impedancePhasor = complex(realImpedance,imagImpedance)
        voltagePhasor = voltageFFT[i]
        
        currentFFT[i] = voltagePhasor/impedancePhasor #V/Z = I
        
    currentSignal = irfft(currentFFT,n=dataLength)
    
    dataframe = pd.DataFrame({'Time (s)':voltageWaveform['Time (s)'],
                              'Current (A)':currentSignal,
                              'Voltage (V)':voltageWaveform['Voltage (V)']})
    chargeArray = sc.integrate.cumulative_trapezoid(dataframe['Current (A)'],
                                                    x = dataframe['Time (s)'])
    chargeArray = np.insert(chargeArray,0,0)
    dataframe['Charge (C)'] = pd.Series(chargeArray)
    dataframe['Time (ms)'] = dataframe['Time (s)'] * 1000
    dataframe['Current Density (mA/cm^2)'] = dataframe['Current (A)']*1000/area
    dataframe['Charge Density (mC/cm^2)'] = dataframe['Charge (C)']*1000/area
        
    return dataframe

#common minimum is 1.258 Hz
baseline = plotCompareNyquist([r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\Gen2_VC\240820_Li_Symm_20um_Gen2+VC_Before cycling_C01.mpt', #1.258 Hz min
                    r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\Gen2_VC\240820_Li_Symm_20um_Gen2+VC_1-1_SingleCycle_C01.mpt', #0.0099853007 Hz min
                    r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\Gen2_VC\240820_Li_Symm_20um_Gen2+VC_1-1_95Cycles_C01.mpt'], #0.0099853007 Hz min
                   'Baseline',
                   [1.2,200000],
                   legendList=['Before Cycling',
                                'Single Cycle',
                                '95 Cycles'],
                   fitModel=True,
                   circuitString='R1-p(CPE2,R2)-p(CPE3,R3-Ws3)',
                   initialGuess=[15 , 5e-6, 1   , 250  , 8e-6   , 1   , 600  , 90000  , 40000  ],
                   bounds=     ([5  , 1e-7, 0.5 , 10   , 1e-6   , 0.5 , 10   , 0      , 0      ],
                                [30 , 9e-5, 1   , 2000 , 5e-5   , 1   , 9000 , 900000 , 100000 ]))
baseline = plotCircuitProperties(baseline,
                      ['Before Cycling',
                       'Single Cycle',
                       '95 Cycles'],
                      saveData=True)
illge = plotCompareNyquist([r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\IL_LGE\Before Cycling\GA240830-1_240902_20um_IL-LGE_BeforeCycling.mpt', #1.258 Hz min
                    r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\IL_LGE\Single Cycling\GA240830-1_240902_20um_IL-LGE_1-1_SingleCycle.mpt', #1.258 Hz min
                    r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\IL_LGE\100 Cycles\GA240830-1_240911_LGE_p20_1_100Cycles_C01.mpt'], #0.0099853007 Hz min
                   'IL-LGE',
                   [1.2,200000],
                   legendList=['Before Cycling',
                                'Single Cycle',
                                '100 Cycles'],
                   fitModel=True,
                   circuitString='R1-p(CPE2,R2)-p(CPE3,R3-Ws3)',
                   initialGuess=[7 , 1e-5, 1   , 250  , 1e-4 , 1   , 250  , 100000  , 40000  ],
                   bounds=     ([3 , 1e-7, 0.5 , 5    , 1e-5 , 0.5 , 10   , 0       , 0      ],
                                [20, 5e-5, 1   , 9000 , 1e-2 , 1   , 9000 , 9000000 , 100000 ]))
illge = plotCircuitProperties(illge,
                      ['Before Cycling',
                        'Single Cycle',
                        '100 Cycles'],
                      saveData=True)

illge.to_csv(r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\DRT_and_Nyquist_Fits\Nyquist_Circuit_Fits\IL-LGE-Components.csv',index=False)
baseline.to_csv(r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\DRT_and_Nyquist_Fits\Nyquist_Circuit_Fits\Baseline-Components.csv',index=False)