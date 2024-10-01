from ReadDataFiles import readPEIS, colorFader, readPEISPandas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from impedance.models.circuits import CustomCircuit
from impedance.preprocessing import cropFrequencies, ignoreBelowX
import os

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

processFolder(r'C:\Users\tejas\Analysis\Potentiostat\Data_Files',[1,7000000])