from ReadDataFiles import readPEIS, colorFader, readRawWaveform, readOSC, convertToImpedanceAnalysis
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from impedance.models.circuits import CustomCircuit
from impedance.preprocessing import cropFrequencies
import os
from pyDRTtools.runs import EIS_object, simple_run
from scipy import fft
import scipy as sc

def plotOneBode(data: pd.DataFrame, title: str):
    """Plots a Bode plot from EIS data.

    Args:
        data (pd.DataFrame): EIS data from readPEIS
        title (str): title of sample

    Returns:
        tuple(matplotlib.figure.Figure,list[matplotlib.axes._axes.Axes]): fig and ax for further customization if necessary
    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xscale('log')
    ax1.plot(data['freq/Hz'],data['|Z|/Ohm'],'k')
    ax1.set_yscale('log')
    ax2.plot(data['freq/Hz'],-data['Phase(Z)/deg'],'r')
    ax1.set(title = title + ' Bode Plot',
           xlabel = 'Frequency (Hz)',
           ylabel = 'Magnitude ($\Omega$)')
    ax2.set(ylabel = '-Phase (deg)')
    
    plt.show()
    
    return (fig, [ax1, ax2])

def plotOneNyquist(eisData: pd.DataFrame, title: str, freqRange: list[float], fitModel: bool = False, circuitString: str = None, initialGuess: list[float] = [], bounds: tuple[list[float]] = ([],[])):
    """Takes EIS data and plots Nyquist.

    Args:
        EISData (pd.DataFrame): EIS Data from ReadDataFiles.readPEIS.
        title (str): title of plot + ' Nyquist Plot'
        freqRange (list[float]): [lower freq bound, upper freq bound]
        fitModel (bool, optional): Whether to fit a circuit model. Defaults to False.
        circuitString (str, optional): Description of circuit model. See impedance.py docs for more details. Defaults to None.
        initialGuess (list[float], optional): Initial guess of circuit model parameters. Defaults to [].
        bounds (tuple[list[float]], optional): Bounds of circuit model parameters (lower bounds, upper bounds). Defaults to ([],[]).

    Returns:
        impedance.models.circuits.CustomCircuit: Fitted circuit. None if fitModel = False.
    """
    fig, ax = plt.subplots()
    f, Z = convertToImpedanceAnalysis(eisData)
    f, Z = cropFrequencies(f,Z,freqRange[0],freqRange[1])
    if fitModel:
        circuit = CustomCircuit(circuitString,initial_guess=initialGuess)
        circuit = circuit.fit(f,Z,bounds)
        zPredict = circuit.predict(f)
        ax.plot(zPredict.real,-zPredict.imag,color='k',linestyle='-',label=circuitString)
    
    ax.plot(Z.real,-Z.imag,'o',color='k')
    ax.set(title = title + ' Nyquist Plot',
           xlabel = 'Re(Z(f)) ($\Omega$)',
           ylabel = '-Im(Z(f)) ($\Omega$)')
    if fitModel == False:
        circuit = None
        
    plt.show()
    
    return circuit

def plotManyNyquists(eisDatas: list[pd.DataFrame], title: str, freqRange: list[float], fitModel: bool = False, circuitString: str = None,initialGuess: list[float] = [],bounds: tuple[list[float]] = ([],[]), legendList: list[str] = None):
    """Takes multiple EIS DataFrames and plots Nyquists + fits.

    Args:
        eisDatas (list[pd.DataFrame]): List of dataframes from ReadDataFiles.readPEIS.
        title (str): Title of samples.
        freqRange (list[float]): [lower freq bound, upper freq bound]
        fitModel (bool, optional): Whether to fit a circuit model. Defaults to False.
        circuitString (str, optional): Description of circuit model. See impedance.py docs for more details. Defaults to None.
        initialGuess (list[float], optional): Initial guess of circuit model parameters. Defaults to [].
        bounds (tuple[list[float]], optional): Bounds of circuit model parameters (lower bounds, upper bounds). Defaults to ([],[]).
        legendList (list[str], optional): List of legend objects. Defaults to None.

    Returns:
        list[impedance.models.circuits.CustomCircuit]: List of fitted circuits. None if fitModel = False.
    """
    numberOfPlots = len(eisDatas)
    circuitList = []
    fig, ax = plt.subplots()
    
    for i in range(0,numberOfPlots):
        
        #gets f and Z values
        f, Z = convertToImpedanceAnalysis(eisDatas[i])
        
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
    
    ax.set(title = title + ' Nyquist Plots',
           xlabel = 'Re(Z(f)) ($\Omega$)',
           ylabel = '-Im(Z(f)) ($\Omega$)')
    
    plt.axis('square')
    
    if legendList != None:
        ax.legend()
    
    plt.show()
    
    return circuitList

def plotCircuitProperties(circuitList: list[CustomCircuit], legendList: list[str]):
    """Takes circuits from plotOneNyquist and plotManyNyquists and plots them in the order of the list. Requires same circuitstring.

    Args:
        circuitList (list[CustomCircuit]): List of fitted circuits.
        legendList (list[str]): List of labels for circuits (x-axis labels)

    Returns:
        pd.DataFrame: circuit properties and errors 
    """
    numberOfCircuits = len(circuitList)
    
    names = circuitList[0].get_param_names()[0]
    units = circuitList[0].get_param_names()[1]
    
    numberOfParameters = len(names)
    
    parameterMatrix = np.zeros((numberOfParameters,numberOfCircuits))
    confMatrix = np.zeros((numberOfParameters,numberOfCircuits))
    
    datadf = pd.DataFrame()
    
    for i in range(0,numberOfParameters):
        for j in range(0,numberOfCircuits):
            parameterMatrix[i,j] = circuitList[j].parameters_[i]
            confMatrix[i,j] = circuitList[j].conf_[i]
            
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

def plotOneDRT():
    """Unfinished function.
    """
    return

def plotManyDRTs(eisDatas: list[pd.DataFrame], title: str, freqRange: list[float], legendList: list[str] = None, rbf_type='Gaussian',der_used='1st order',cv_type='GCV',reg_param=1E-4,shape_control='FWHM Coefficient',coeff=0.1):
    """Calculates non-Bayesian DRT. May require extensive hyperparameter tuning. Use bayesDRT.py instead.

    Args:
        eisDatas (list[pd.DataFrame]): EIS data in pandas dataframe.
        title (str): Title of plot.
        freqRange (list[float]): [lower freq bound, upper freq bound]
        legendList (list[str], optional): List of legend items. Defaults to None.
        rbf_type (str, optional): See bayesdrt2 docs. Defaults to 'Gaussian'.
        der_used (str, optional): See bayesdrt2 docs. Defaults to '1st order'.
        cv_type (str, optional): See bayesdrt2 docs. Defaults to 'GCV'.
        reg_param (_type_, optional): See bayesdrt2 docs. Defaults to 1E-4.
        shape_control (str, optional): See bayesdrt2 docs. Defaults to 'FWHM Coefficient'.
        coeff (float, optional): See bayesdrt2 docs. Defaults to 0.1.

    Returns:
        tuple(matplotlib.figure.Figure,matplotlib.axes._axes.Axes): fig and ax for further customization if necessary
    """
    numberOfPlots = len(eisDatas)
    fig, ax = plt.subplots()
    ax.set_title(title)
    
    for i in range(0,numberOfPlots):
        
        data = eisDatas[i]
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
        ax.plot(data.out_tau_vec, data.gamma,color=color)
    
    ax.set(ylabel = r'$\gamma$ ($\Omega$)',
           xlabel = r'RC Time Constant (s)',
           xscale='log')
    if legendList != None:
        ax.legend(legendList)
        
    plt.show()
    
    return (fig, ax)

def generateCircuitFit(f: np.ndarray[float], Z: np.ndarray[complex], circuitString: str, initialGuess: list[float]):
    """Takes in data from convertToImpedanceAnalysis and fits it to a specific circuit.

    Args:
        f (np.ndarray[float]): frequencies
        Z (np.ndarray[complex]): impedances
        circuitString (str): string of circuit, see impedance.py docs
        initialGuess (list[float]): initial guesses for circuit parameters
        
    Returns:
        circuit (CustomCircuit): can use this to plot
    """
    #generates circuit model
    
    circuit = CustomCircuit(circuitString,initial_guess=initialGuess)
    circuit = circuit.fit(f,Z)
    ZValues1 = circuit.predict(f)
    
    fig, ax = plt.subplots()
    ax.plot(Z.real,-Z.imag,'ko')
    ax.plot(ZValues1.real,-ZValues1.imag,'r-')
    
    plt.show()
    
    return circuit

def predictCurrent(peisFilename,voltageWaveformFilename,pH,area,referencePotential):
    """Experimental function to use EIS impedance to predict current waveform from voltage waveform.

    Args:
        peisFilename (_type_): _description_
        voltageWaveformFilename (_type_): _description_
        pH (_type_): _description_
        area (_type_): _description_
        referencePotential (_type_): _description_

    Returns:
        _type_: _description_
    """
    #sets DC offset so 0 current is HER equilibrium
    DCOffset = referencePotential + 0.059*pH
    
    #use only seconds and hertz for units to work out
    peisData = readPEIS(peisFilename)
    peisData = peisData.sort_values('freq/Hz')
    if 'f_' in voltageWaveformFilename:
        voltageWaveform = readRawWaveform(voltageWaveformFilename,pH,referencePotential)
    else:
        voltageWaveform = readOSC(voltageWaveformFilename,pH,area,referencePotential,'1 A')
    voltageWaveform['RawVoltage (V)'] = voltageWaveform['RawVoltage (V)'] + DCOffset
    dataLength = voltageWaveform.shape[0]
    
    dt = voltageWaveform['Time (s)'].diff().mean()
    
    voltageFFT = fft.rfft(voltageWaveform['RawVoltage (V)'].to_numpy())
    voltageFFTFreq = fft.rfftfreq(dataLength, dt)
    currentFFT = np.zeros(voltageFFT.size,dtype=complex)
    
    for i in range(len(voltageFFT)):
        #linearly interpolates peisData to find best value based on frequency
        imagImpedance = -np.interp(voltageFFTFreq[i],peisData['freq/Hz'],peisData['-Im(Z)/Ohm'])
        realImpedance = np.interp(voltageFFTFreq[i],peisData['freq/Hz'],peisData['Re(Z)/Ohm'])
        impedancePhasor = complex(realImpedance,imagImpedance)
        voltagePhasor = voltageFFT[i]
        
        currentFFT[i] = voltagePhasor/impedancePhasor #V/Z = I
        
    currentSignal = fft.irfft(currentFFT,n=dataLength)
    
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

def getEISFromDynamicCA(filename, pH, area, referencePotential, irange, fundamental_freq_hz, cutoff, bode=False, stretch=1):
    """
    Calculate EIS from oscilloscope data. Experimental.
    
    Parameters:
    -----------
    filename : str
        .csv file from oscilloscope
    pH : float
        pH of solution for RHE conversation
    area : float
        area of electrode in cm^2
    referencePotential : float
        potential of reference electrode
    irange : str
        acceptable is '2A', '1A', '100mA', '10mA'
    fundamental_freq_hz : float
        Frequency of the excitation waveform in Hz
    cutoff : float
        Cutoff frequency (see Bode plot to judge when noise becomes excessive)
    bode (optional) : bool
        Whether to plot bode plot or not
    stretch (optional) : int
        For reading waveform (see readOSC())
        
    Returns:
    --------
    output : pd.DataFrame('freq/Hz','Re(Z)/Ohm','-Im(Z)/Ohm')
    """
    
    
    df = readOSC(filename,pH,area,referencePotential,irange,stretch=stretch)
    
    # Get data
    t = df['Time (s)'].values
    v = df['Voltage (V)'].values
    i = df['Current (A)'].values
    
    # Calculate sampling parameters
    sample_rate = 1 / (t[1] - t[0])
    total_time = t[-1] - t[0]
    
    # Calculate num_periods properly from actual time duration
    num_periods = total_time * fundamental_freq_hz
    print(f"Number of periods: {num_periods:.2f}")
    
    # Compute FFTs and normalize
    v_fft = fft.rfft(v) / len(v)
    i_fft = fft.rfft(i) / len(i)
    
    # Calculate impedance
    Z = v_fft / i_fft
    
    # Generate frequency array
    freqs = fft.rfftfreq(len(v), d=t[1]-t[0])
    
    # Take only positive frequencies up to cutoff
    mask = (freqs > fundamental_freq_hz) & (freqs < cutoff)
    freqs = freqs[mask]
    Z = Z[mask]
    
    #converts to PEIS Pandas format
    output = pd.DataFrame()
    
    output['freq/Hz'] = freqs
    output['Re(Z)/Ohm'] = Z.real
    output['-Im(Z)/Ohm'] = -Z.imag
    
    if bode:
        #plots Bode plot
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        ax1.semilogx(freqs, 20 * np.log10(np.abs(Z)),'k')
        ax1.set_ylabel('|Z| (dB Ω)')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.grid(True)
        
        ax2.semilogx(freqs, -np.angle(Z, deg=True),'r')
        ax2.set_ylabel('-Phase (degrees)')
        ax2.grid(True)
        plt.show()
        
    #saves pandas dataframe
        
    return output