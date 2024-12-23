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
        color = colorFader('blue','red',i,numberOfPlots)
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

def plotCompareDRT(filenames,title,freqRange,legendList=None):
    
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
                        rbf_type='Gaussian',
                        data_used='Combined Re-Im Data',
                        induct_used=0,
                        der_used='1st order',
                        cv_type='GCV',
                        reg_param= 1E-3, #1E-4 is good
                        shape_control='FWHM Coefficient',
                        coeff=0.1) #0.3 is good
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

# initialEIS = predictCurrent(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\6_PEIS_HER_After_Debubbling_02_PEIS_C01.txt',
#                     r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\11_100kHz_PW-6_Dynamic_CA.csv',
#                     14,
#                     0.182,
#                     0.209)
# finalEIS = predictCurrent(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\15_PEIS_HER_02_PEIS_C01.txt',
#                     r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\11_100kHz_PW-6_Dynamic_CA.csv',
#                     14,
#                     0.182,
#                     0.209)
# realData = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\11_100kHz_PW-6_Dynamic_CA.csv',
#                    14,
#                    0.182,
#                    0.209,
#                    '1 A')
# plotWaveforms([initialEIS,finalEIS,realData],'Comparison',['Initial EIS','Final EIS','Real'],True)

# pwn6 = predictCurrent(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\15_PEIS_HER_02_PEIS_C01.txt',
#                       r'C:\Users\tejas\Analysis\Potentiostat\Waveforms\f_1d000000Ep03_PW_1d000En06_F_n1d37_U_p5d5_D_n3d0_False.csv',
#                       14,
#                       0.1826403875,
#                       0.209)
# pwn5 = predictCurrent(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\15_PEIS_HER_02_PEIS_C01.txt',
#                       r'C:\Users\tejas\Analysis\Potentiostat\Waveforms\f_1d000000Ep03_PW_1d000En05_F_n1d37_U_p5d5_D_n3d0_False.csv',
#                       14,
#                       0.1826403875,
#                       0.209)
# pwn4 = predictCurrent(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\15_PEIS_HER_02_PEIS_C01.txt',
#                       r'C:\Users\tejas\Analysis\Potentiostat\Waveforms\f_1d000000Ep03_PW_1d000En04_F_n1d37_U_p5d5_D_n3d0.csv',
#                       14,
#                       0.1826403875,
#                       0.209)
# realpwn6 = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\8_1000Hz_PW-6_Dynamic_CA.csv',
#                    14,
#                    0.1826403875,
#                    0.209,
#                    '1 A')
# realpwn5 = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\7_1000Hz_PW-5_Dynamic_CA.csv',
#                    14,
#                    0.1826403875,
#                    0.209,
#                    '1 A')
# realpwn4 = readOSC(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-01-TN-01-051\9_1000Hz_Dynamic_CA.csv',
#                    14,
#                    0.1826403875,
#                    0.209,
#                    '1 A')

# plotWaveforms([pwn6,pwn5,pwn4,realpwn6,realpwn5,realpwn4],
#               'Argon-Saturated Current Responses',
#               ['1 us Ideal',
#                '10 us Ideal',
#                '100 us Ideal',
#                '1 us Real',
#                '10 us Real',
#                '100 us Real'],
#               True,
#               customColors = ['#FF5F5F',
#                               '#FF1E1E',
#                               '#7A0000',
#                               '#88B2ED',
#                               '#4889CD',
#                               '#183859'])

# plotCompareDRT([#r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-07-31-TN-01-050\6_PEIS_HER_02_PEIS_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-07-31-TN-01-050\18_PEIS_HER_02_PEIS_C01.txt',
#                 #r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-05-TN-01-053\6_PEIS_HER_afterdebubbling_02_PEIS_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-05-TN-01-053\16_PEIS_HER_02_PEIS_C01.txt',
#                 #r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-01-TN-01-051\5_PEIS_HER_02_PEIS_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-01-TN-01-051\15_PEIS_HER_02_PEIS_C01.txt',
#                 #r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\6_PEIS_HER_After_Debubbling_02_PEIS_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-08-04-TN-01-052\15_PEIS_HER_02_PEIS_C01.txt'
#                 ],
#                'Ar vs. N2 Distribution of Relaxation Times',
#                [1,200000],
#                ['Initial N2',
#                 'Final N2',
#                 'Initial Ar',
#                 'Final Ar'])

# plotCompareNyquist([r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-04\Tafel-Ultra-tin-SRO-BTO-SRO-NSTO-1M KOH-Graphite Counter- SCE Ref-continous_17_PEIS_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-04\Tafel-3UC-SRO-BTO-NSTO-1M KOH-Graphite Counter- SCE Ref-continous_17_PEIS_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-04\Tafel-Thick SRO-NSTO-1M KOH-Graphite Counter- SCE Ref-Repeat2_17_PEIS_C01.txt'],
#                    '',
#                [1,200000],
#                True,
#                circuitString='p(p(R1,CPE1),p(R2,CPE2))-R0',
#                bounds = ([0,0,0,0,0,0,0],
#                          [8000,1000e-6,1,8000,1000e-6,1,70]),
#                initialGuess=[1000,50e-6,1,1000,50e-6,1,16],
#                legendList=['1.5 UC SRO',
#                 '3 UC SRO',
#                 '48 UC SRO'])
# plotCompareDRT([r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-04\Tafel-Ultra-tin-SRO-BTO-SRO-NSTO-1M KOH-Graphite Counter- SCE Ref-continous_17_PEIS_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-04\Tafel-3UC-SRO-BTO-NSTO-1M KOH-Graphite Counter- SCE Ref-continous_17_PEIS_C01.txt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Pedram_Archive\2022-08-04\Tafel-Thick SRO-NSTO-1M KOH-Graphite Counter- SCE Ref-Repeat2_17_PEIS_C01.txt'],
#                'Comparison',
#                [1,200000],
#                ['1.5 UC SRO',
#                 '3 UC SRO',
#                 '48 UC SRO'])

# plotCompareDRT([r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\IL_LGE\Before Cycling\GA240830-1_240902_20um_IL-LGE_BeforeCycling.mpt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\IL_LGE\Single Cycling\GA240830-1_240902_20um_IL-LGE_1-1_SingleCycle.mpt',
#                 r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\IL_LGE\100 Cycles\GA240830-1_240911_LGE_p20_1_100Cycles_C01.mpt'],
#                'DRT Comparison IL-LGE',
#                [5,1.5E6],
#                ['Before Cycling',
#                 'Single Cycle',
#                 '100 Cycles'])
# plotCompareNyquist([r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\IL_LGE\Before Cycling\GA240830-1_240902_20um_IL-LGE_BeforeCycling.mpt',
#                     r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\IL_LGE\Single Cycling\GA240830-1_240902_20um_IL-LGE_1-1_SingleCycle.mpt',
#                     r'C:\Users\tejas\Analysis\Potentiostat\Ganesh Files\IL_LGE\100 Cycles\GA240830-1_240911_LGE_p20_1_100Cycles_C01.mpt'],
#                    'IL-LGE',
#                    [5,1.5E6],
#                    legendList=['Before Cycling',
#                                 'Single Cycle',
#                                 '100 Cycles'],
#                    fitModel=True,
#                    circuitString='R1-p(C2,R2)-p(C3,R3)',
#                    initialGuess=[7,1e-6,30,1e-6,400],
#                    bounds=([5 ,1e-8,25 ,1e-8,20 ],
#                            [15,1e-2,377,1e-2,500]))

plotCompareDRT([#r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\9_PEIS_HER_Down_01_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\9_PEIS_HER_Down_02_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\9_PEIS_HER_Down_03_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\9_PEIS_HER_Down_04_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\9_PEIS_HER_Down_05_PEIS_C01.txt',],
               'Poled Down',
               [1,2E5],
               [#'0 $V_{RHE}$',
                '-0.1 $V_{RHE}$',
                '-0.2 $V_{RHE}$',
                '-0.3 $V_{RHE}$',
                '-0.4 $V_{RHE}$'])
plotCompareNyquist([#r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\9_PEIS_HER_Down_01_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\9_PEIS_HER_Down_02_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\9_PEIS_HER_Down_03_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\9_PEIS_HER_Down_04_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\9_PEIS_HER_Down_05_PEIS_C01.txt',],
                   'Poled Down',
                   [1,2E5],
                   legendList=[#'0 $V_{RHE}$',
                '-0.1 $V_{RHE}$',
                '-0.2 $V_{RHE}$',
                '-0.3 $V_{RHE}$',
                '-0.4 $V_{RHE}$'])
plotCompareDRT([#r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\11_PEIS_HER_Up_01_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\11_PEIS_HER_Up_02_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\11_PEIS_HER_Up_03_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\11_PEIS_HER_Up_04_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\11_PEIS_HER_Up_05_PEIS_C01.txt',],
               'Poled Up',
               [1,2E5],
               [#'0 $V_{RHE}$',
                '-0.1 $V_{RHE}$',
                '-0.2 $V_{RHE}$',
                '-0.3 $V_{RHE}$',
                '-0.4 $V_{RHE}$'])
plotCompareNyquist([#r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\11_PEIS_HER_Up_01_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\11_PEIS_HER_Up_02_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\11_PEIS_HER_Up_03_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\11_PEIS_HER_Up_04_PEIS_C01.txt',
                r'C:\Users\tejas\Analysis\Potentiostat\Data_Files\2024-10-27-TN-01-056\11_PEIS_HER_Up_05_PEIS_C01.txt',],
                   'Poled Up',
                   [1,2E5],
                   legendList=[#'0 $V_{RHE}$',
                '-0.1 $V_{RHE}$',
                '-0.2 $V_{RHE}$',
                '-0.3 $V_{RHE}$',
                '-0.4 $V_{RHE}$'])