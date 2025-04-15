import pandas as pd
import numpy as np
import matplotlib as mpl
import scipy as sc
import re


def readCV(filename: str, pH: float, area: float, referencePotential: float): #area is in cm^2
    """Reads cyclic voltammetry data from Biologic into Pandas dataframe.

    Args:
        filename (str): filename of CV .txt from biologic (must be exported using "CV-all") or .mpt file
        pH (float): pH of electrolyte for RHE conversion
        area (float): geometric area of electrode in cm^2 for current density
        referencePotential (float): potential of reference electrode vs. SHE in V

    Returns:
        pd.DataFrame: dataframe of CV data
    """
    if filename[-3:] == 'mpt':
        
        #finds number of header lines
        with open(filename, 'r') as file:
            file.readline()
            line = file.readline()
            numHeaderLines = int(re.findall(r'-?\d*\.?\d+', line)[0])
            
        data = pd.read_csv(filename,
                        sep='\s+',
                        skiprows=numHeaderLines,
                        names = ['mode',
                                'ox/red',
                                'error',
                                'control changes',
                                'counter inc.',
                                'time/s',
                                'control/V',
                                'Ewe/V',
                                'Ece/V',
                                'I/mA',
                                'cycle number',
                                '(Q-Qo)/C',
                                'I Range',
                                'Analog IN 2/V',
                                'Rcmp/Ohm',
                                'step time/s',
                                'Pwe/W',
                                'Pce/W',
                                'Pwe-ce/W',
                                'Ewe-Ece/V',
                                'Rew/Ohm',
                                'Rce/Ohm',
                                'Rwe-ce/Ohm'],
                        index_col=False,
                        dtype = np.float64,
                        encoding='windows-1252')
        
    else:
    
        data = pd.read_csv(filename,
                        sep='\s+',
                        skiprows=1,
                        names = ['mode',
                                    'ox/red',
                                    'error',
                                    'control changes',
                                    'counter inc.',
                                    'time/s',
                                    'control/V',
                                    'Ewe/V',
                                    'I/mA',
                                    'cycle number',
                                    '(Q-Qo)/C',
                                    'I Range',
                                    '<Ece>/V',
                                    'Analog IN 2/V',
                                    'Rcmp/Ohm',
                                    'P/W',
                                    'Ewe-Ece/V'],
                        index_col=False,
                        dtype = np.float64,
                        engine='python')
    
    data['I/A'] = data['I/mA']/1000
    data['j/mA*cm-2'] = data['I/mA']/area
    data['Ewe/V'] = data['Ewe/V'] + referencePotential + 0.059*pH
    data['Ewe/mV'] = data['Ewe/V']*1000
    data['j/A*m-2'] = data['j/mA*cm-2']*10
    
    #cleans data to remove points where Synology interferes with saving data
    data = data[~((data['time/s'] == 0) & (data['I/mA'] == 0))]
    
    return data

def readCA(filename: str, pH: float, area: float, referencePotential: float, shouldRemoveNoise: bool = False): #area is in cm^2
    """Reads chronoamperometry data from Biologic into pandas dataframe.

    Args:
        filename (str): filename of CV .txt from biologic (must be exported using "CV-all") or .mpt file
        pH (float): pH of electrolyte for RHE conversion
        area (float): geometric area of electrode in cm^2 for current density
        referencePotential (float): potential of reference electrode vs. SHE in V
        shouldRemoveNoise (bool): if true, will remove standard noise using ReadDataFiles.removeNoise

    Returns:
        pd.DataFrame: dataframe of CA data
    """
    
    if filename[-3:] == 'mpt':
        
        #finds number of header lines
        with open(filename, 'r') as file:
            file.readline()
            line = file.readline()
            numHeaderLines = int(re.findall(r'-?\d*\.?\d+', line)[0])
            
        data = pd.read_csv(filename,
                            sep='\s+',
                            skiprows=numHeaderLines,
                            names = ['mode',
                                    'ox/red',
                                    'error',
                                    'control changes',
                                    'Ns changes',
                                    'counter inc.',
                                    'Ns',
                                    'time/s',
                                    'control/V',
                                    'Ewe/V',
                                    'Ece/V',
                                    'I/mA',
                                    'dQ/C',
                                    '(Q-Qo)/C',
                                    'I range',
                                    'Q charge/discharge/mA.h',
                                    'half cycle',
                                    'Analog IN 2/V',
                                    'Energy we charge/W.h',
                                    'Energy we discharge/W.h',
                                    'Energy ce charge/W.h',
                                    'Energy ce discharge/W.h',
                                    'Capacitance charge/muF',
                                    'Capacitance discharge/muF',
                                    'step time/s',
                                    'Q discharge/mA.h',
                                    'Q charge/mA.h',
                                    'Capacity/mA.h',
                                    'Efficiency/%',
                                    'cycle number',
                                    'Pwe/W',
                                    'Pce/W',
                                    'Pwe-ce/W',
                                    'Ewe-Ece/V',
                                    'Rwe/Ohm',
                                    'Rce/Ohm',
                                    'Rwe-ce/Ohm',
                                    'Energy we-ce charge/W.h',
                                    'Energy we-ce dischage/W.h'],
                            index_col=False,
                            dtype = np.float64,
                            encoding='windows-1252')
            
    else:
    
        data = pd.read_csv(filename,
                        sep='\s+',
                        skiprows=1,
                        names = ['mode',
                                    'ox/red',
                                    'error',
                                    'control changes',
                                    'Ns changes',
                                    'counter inc.',
                                    'Ns',
                                    'time/s',
                                    'control/V',
                                    'Ewe/V',
                                    'I/mA',
                                    'dQ/C',
                                    'I range',
                                    'Ece/V',
                                    'Analog IN 2/V',
                                    'Rcmp/Ohm',
                                    'Capacitance charge/muF',
                                    'Capacitance discharge/muF',
                                    'Efficiency/%',
                                    'cycle number',
                                    'P/W'],
                        index_col=False,
                        dtype = np.float64,
                        encoding='unicode_escape')
    
    if shouldRemoveNoise:
        data = removeNoise(data)

    data['j/mA*cm-2'] = data['I/mA']/area
    data['Ewe/V'] = data['Ewe/V'] + referencePotential + 0.059*pH
    data['Ewe/mV'] = data['Ewe/V']*1000
    data['j/A*m-2'] = data['j/mA*cm-2']*10
    
    #cleans data to remove points where Synology interferes with saving data
    data = data[~((data['time/s'] == 0) & (data['I/mA'] == 0))]
    
    return data

def readPEIS(filename: str):
    """Reads potentiostatic electrochemical impedance spectroscopy data from Biologic into pandas dataframe.

    Args:
        filename (str): filename of PEIS .txt (must be exported using PEIS-all) or PEIS .mpt

    Returns:
        pd.DataFrame: dataframe of PEIS data
    """
    if filename[-3:] == 'mpt':
        
        #finds number of header lines
        with open(filename, 'r') as file:
            file.readline()
            line = file.readline()
            numHeaderLines = int(re.findall(r'-?\d*\.?\d+', line)[0])
            
        data = pd.read_csv(filename,
                        sep='\s+',
                        skiprows=numHeaderLines,
                        names = ['freq/Hz',
                                'Re(Z)/Ohm',
                                '-Im(Z)/Ohm',
                                '|Z|/Ohm',
                                'Phase(Z)/deg',
                                'time/s',
                                '<Ewe>/V',
                                '<I>/mA',
                                'Cs/uF',
                                'Cp/uF',
                                'cycle number',
                                'I Range',
                                '|Ewe|/V',
                                '|I|/A',
                                'Ns',
                                '(Q-Qo)/mA.h',
                                '<Ece>/V',
                                '|Ece|/V',
                                'Phase(Zce)/deg',
                                '|Zce|/Ohm',
                                'Re(Zce)/Ohm',
                                '-Im(Zce)/Ohm',
                                'Phase(Zwe-ce)/deg',
                                '|Zwe-ce|/Ohm',
                                'Re(Zwe-ce)/Ohm',
                                '-Im(Zwe-ce)/Ohm',
                                'Analog IN 2/V',
                                'Re(Y)/Ohm-1',
                                'Im(Y)/Ohm-1',
                                '|Y|/Ohm-1',
                                'Phase(Y)/deg',
                                'Re(C)/nF',
                                'Im(C)/nF',
                                '|C|/nF',
                                'Phase(C)/deg',
                                'Re(M)',
                                'Im(M)',
                                '|M|',
                                'Phase(M)/deg',
                                'Re(Permittivity)',
                                'Im(Permittivity)',
                                '|Permittivity|',
                                'Phase(Permittivity)/deg',
                                'Re(Resistivity)/Ohm.cm',
                                'Im(Resistivity)/Ohm.cm',
                                '|Resistivity|/Ohm.cm',
                                'Phase(Resistivity)/deg',
                                'Re(Conductivity)/mS/cm',
                                'Im(Conductivity)/mS/cm',
                                '|Conductivity|/mS/cm',
                                'Phase(Conductivity)/deg',
                                'Tan(Delta)',
                                'Loss Angle(Delta)/deg',
                                'dq/mA.h',
                                'Pwe/W',
                                'Pce/W',
                                'Pwe-ce/W',
                                '<Ewe-Ece>/V',
                                'Rwe/Ohm',
                                'Rce/Ohm',
                                'Rwe-ce/Ohm',
                                'Ewe-Ece/V'],
                        index_col=False,
                        dtype = np.float64,
                        encoding='windows-1252')
    else:
        data = pd.read_csv(filename,
                        sep='\s+',
                        skiprows=1,
                        names = ['freq/Hz',
                                    'Re(Z)/Ohm',
                                    '-Im(Z)/Ohm',
                                    '|Z|/Ohm',
                                    'Phase(Z)/deg',
                                    'time/s',
                                    '<Ewe>/V',
                                    '<I>/mA',
                                    'Cs/uF',
                                    'Cp/uF',
                                    'cycle number',
                                    'I Range',
                                    '|Ewe|/V',
                                    '|I|/A',
                                    'Ns',
                                    '(Q-Qo)/mA.h',
                                    'Re(Y)/Ohm-1',
                                    'Im(Y)/Ohm-1',
                                    '|Y|/Ohm-1',
                                    'Phase(Y)/deg',
                                    'dq/mA.h'],
                        index_col=False,
                        dtype = np.float64,
                        encoding='unicode_escape')
        
    #cleans data to remove points where Synology interferes with saving data
    data = data[~((data['time/s'] == 0) & (data['<I>/mA'] == 0))]
    
    return data

def readPEISImpedance(filename: str):
    """Reads PEIS directly into format that impedance.py can use.

    Args:
        filename (str): filename of PEIS .txt (must be exported using PEIS-all) or PEIS .mpt

    Returns:
        tuple(np.ndarray(float),np.ndarray(complex)): (frequency, complex impedance)
    """
    return convertToImpedanceAnalysis(readPEIS(filename))

def convertToImpedanceAnalysis(data: pd.DataFrame):
    """Converts to format that impedance.py can use.

    Args:
        data (pd.DataFrame): Output of ReadDataFiles.readPEIS.

    Returns:
        tuple(np.ndarray(float),np.ndarray(complex)): (frequency, complex impedance)
    """
    frequency = data['freq/Hz'].to_numpy()
    dataLength = len(frequency)
    realImpedance = data['Re(Z)/Ohm'].to_numpy()
    imagImpedance = -data['-Im(Z)/Ohm'].to_numpy()
    impedance = np.zeros(dataLength,dtype=np.complex_)
    
    for i in range(0,dataLength):
        impedance[i] = complex(realImpedance[i],
                               imagImpedance[i])
    
    return (frequency,impedance)

def readOSC(filename: str,pH: float, area: float, referencePotential: float, irange: str, stretch: float = 1):
    """Reads oscilloscope .csv from Picoscope.

    Args:
        filename (str): filename of waveform .csv from Picoscope
        pH (float): pH of electrolyte for RHE conversion
        area (float): geometric area of electrode in cm^2 for current density
        referencePotential (float): potential of reference electrode vs. SHE in V
        irange (str): irange of measurement; '2A', '1A', '100mA', '10mA' are acceptable
        stretch (float, optional): For 1 Hz, typically 4, for 10 Hz, typically 2. Defaults to 1.

    Returns:
        pd.DataFrame: dataframe containing oscilloscope values
    """
    with open(filename,'r') as file:
        lines = file.readlines()
        firstChannel = lines[0][33]
        unitsList = lines[1].split(',')
        unitsList = [i.replace('(','').replace(')','') for i in unitsList]
        timeUnit = unitsList[0]
        voltage1Unit = unitsList[3]
        voltage2Unit = unitsList[4]
        
    names = ['Time (xs)','discard1','discard2','ch1Avg','ch2Avg']
    
    data = pd.read_csv(filename,
                       skiprows=3,
                       sep=',',
                       names=names,
                       index_col=False,
                       dtype=float)
    
    #ensures correct units
    if voltage1Unit == 'mV':
        data['ch1Avg'] = data['ch1Avg'] / 1000
    if voltage2Unit == 'mV':
        data['ch2Avg'] = data['ch2Avg'] / 1000
    
    #ensures time starts at 0
    data['Time (xs)'] = data['Time (xs)'] - data['Time (xs)'].loc[0]
    
    #sets proper units of time
    if timeUnit == 'ms':
        data['Time (ms)'] = data['Time (xs)']
    elif timeUnit == 'us':
        data['Time (ms)'] = data['Time (xs)']/1000
        
    #ensures correct labels
    if firstChannel == 'A':
        data['Voltage (V)'] = data['ch1Avg']
        data['Current (A)'] = data['ch2Avg']
    else:
        data['Voltage (V)'] = data['ch2Avg']
        data['Current (A)'] = data['ch1Avg']
    
    data['Time (ms)'] *= stretch
    data = data[data['Time (ms)'] < (data['Time (ms)'].max()/stretch)]
    data['Time (s)'] = data['Time (ms)']/1000
    data['RawVoltage (V)'] = data['Voltage (V)']
    data['Voltage (V)'] = data['Voltage (V)'] + referencePotential + 0.059*pH
    
    if irange == '1A' or irange == '2A': #unsure if this is right
        data['Current (A)'] = data['Current (A)']
    elif irange == '100mA':
        data['Current (A)'] = data['Current (A)']*0.1
    elif irange == '10mA':
        data['Current (A)'] = data['Current (A)']*0.01
    
    chargeArray = sc.integrate.cumulative_trapezoid(data['Current (A)'],
                                                    x = data['Time (s)'])
    chargeArray = np.insert(chargeArray,0,0)
    data['Charge (C)'] = pd.Series(chargeArray)
    data['Charge Density (mC/cm^2)'] = data['Charge (C)']*1000/area
    
    data['Current (mA)'] = data['Current (A)']*1000
    data['Current Density (mA/cm^2)'] = data['Current (mA)']/area
    
    data = data.drop(['discard1','discard2','Time (xs)','ch1Avg','ch2Avg'],axis=1)

    return data

def readRawWaveform(filename: str, pH: float, referencePotential: float):
    """Reads waveforms that Siglent can read into a Pandas DataFrame.

    Args:
        filename (str): filename of waveform .csv from makeWaveform.py or from Siglent software.
        pH (float): pH of electrolyte for RHE conversion
        referencePotential (float): potential of reference electrode vs. SHE in V

    Returns:
        pd.DataFrame: dataframe with 'Time (s)', 'Time (ms)', 'RawVoltage (V)', 'Voltage (V)'
    """
    with open(filename,'r') as file:
        lines = file.readlines()
        dataLength = float(lines[0][12:])
        frequency = float(lines[1][10:])
    
    data = pd.read_csv(filename,
                       skiprows=12,
                       sep=',',
                       dtype=float)
    data['Time (s)'] = (data['xpos']-1)/(frequency*dataLength)
    data['Time (ms)'] = data['Time (s)']*1000
    data['RawVoltage (V)'] = data['value']
    data['Voltage (V)'] = data['value'] + referencePotential + 0.059*pH
    
    return data

def readDRT(filename: str):
    """Reads DRT generated by pyDRTTools.

    Args:
        filename (str): filename of DRT .csv

    Returns:
        pd.DataFrame: dataframe with 'tau', 'gamma'
    """
    data = pd.read_csv(filename,
                       skiprows=2,
                       sep=',',
                       dtype=float)
    return data

def colorFader(c1: str,c2: str,currentIndex: int, totalIndices: int):
    """Generates color gradient for plotting.

    Args:
        c1 (str): color of first index
        c2 (str): color of last index
        currentIndex (int): index to linearly interpolate between colors (1 is c1, totalIndices is c2)
        totalIndices (int): total number of indices

    Returns:
        color: matplotlib color in hex format
    """
    if totalIndices > 1:
        mix = currentIndex/(totalIndices-1)
    else:
        mix = 0
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def removeNoise(df: pd.DataFrame, fundamental_freq: float = 0.46, Q: float =1):
    """
    Apply multiple notch filters to remove a fundamental frequency and its harmonics
    from an electrochemical signal.
    
    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame containing at least 'time/s' and 'I/mA' columns
    fundamental_freq : float, optional
        The fundamental frequency to remove in Hz, default is 0.46 Hz
    Q : float, optional
        Quality factor for the notch filters. Higher values create narrower notches.
        Default is 30.
        
    Returns:
    -------
    pandas.DataFrame
        A copy of the input DataFrame with filtered 'I/mA' values
    """
    # Make a copy of the input DataFrame to avoid modifying the original
    filtered_df = df.copy()
    
    # Extract the time and current data
    time = df['time/s'].values
    current = df['I/mA'].values
    
    # Calculate sampling frequency from the time data
    # Using median difference to handle potential irregularities in sampling
    time_diffs = np.diff(time)
    dt = np.median(time_diffs)
    fs = 1/dt
    
    # Apply notch filter for fundamental and each harmonic
    filtered_current = current.copy()
    
    frequencies_removed = []
    
    belowNyquistFreq = True
    i = 1
    while belowNyquistFreq:
        target_freq = i * fundamental_freq
        
        # Skip if the frequency is above Nyquist frequency
        if target_freq > fs/2:
            belowNyquistFreq = False
            break
        
        # Create and apply the notch filter
        b, a = sc.signal.iirnotch(target_freq, Q, fs)
        filtered_current = sc.signal.filtfilt(b, a, filtered_current)
        
        frequencies_removed.append(target_freq)
        i += 1
    
    # Update the DataFrame with the filtered data
    filtered_df['I/mA'] = filtered_current
    
    return filtered_df

def calculateIntegral(timeSeries: pd.Series, valueSeries: pd.Series, baseline: float, timeBounds: list[float]):
    """Calculates 1D integral of (valueSeries-baseline) over timeSeries bounded by timeBounds.

    Args:
        timeSeries (pd.Series): x-values of integration
        valueSeries (pd.Series): y-values of integration
        baseline (float): amount to subtract valueSeries by
        timeBounds (list[float]): [0] is lower bound, [1] is upper bound

    Returns:
        float: value of integral
    """
    #truncates timeSeries and valueSeries according to timeBounds
    timeSeries = timeSeries[(timeSeries >= timeBounds[0]) & (timeSeries <= timeBounds[1])]
    valueSeries = valueSeries.loc[timeSeries.first_valid_index():timeSeries.last_valid_index()]
    
    #subtracts baseline from valueSeries
    valueSeries = valueSeries - baseline
    
    #integrates
    return sc.integrate.trapezoid(y=valueSeries,x=timeSeries)

def readBayesDRT2(filename: str, freqRange: list[float]):
    """Reads .txt or .mpt file for PEIS from Biologic into freq, Z for bayesDRT2 library

    Args:
        filename (str): filename of PEIS .txt (must be exported using PEIS-all) or PEIS .mpt
        freqRange (list[float]): [lowerFreq,upperFreq] defines frequency range to read file over

    Returns:
        tuple(np.ndarray(float),np.ndarray(complex)): (frequency, complex impedance)
    """
    data = readPEIS(filename)
    data = data[(data['freq/Hz'] >= freqRange[0]) & (data['freq/Hz'] <= freqRange[1])]
    freq = data['freq/Hz'].values
    Z = data['Re(Z)/Ohm'].values - 1j * data['-Im(Z)/Ohm'].values
    
    return freq, Z

def readExcelSheet(filename: str):
    """Gets H2 produced from Excel sheet. Ensure that the Excel sheet contains the correct analysis
    and that the Excel sheet names start with the experiment number.

    Args:
        filename (str): .xlsx file with GC data
        
    Returns:
        dict[experimentNumber] = (H2 Value (mol), H2 Error (mol)): {int,(np.float64,np.float64)}
    """
    
    finalDict = {}
    
    xl = pd.ExcelFile(filename)

    for sheet_name in xl.sheet_names:
        
        try:
            experimentNumber = int(re.search(r'\d+', str(sheet_name)).group())
        except:
            continue
        
        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)
        
        # Search for the cell in this sheet
        foundValue = False
        for row_idx, row in df.iterrows():
            for col_idx, value in enumerate(row):
                if isinstance(value, str) and "Total H2 Generated (mol)" in value:
                    h2prod = np.float64(df.iloc[row_idx, col_idx + 1])
                    h2err = np.float64(df.iloc[row_idx, col_idx + 2])
                    foundValue = True
                    break
            if foundValue:
                break
            
        finalDict[experimentNumber] = (h2prod,h2err)
    
    return finalDict

def readOCV(filename: str, pH: float, referencePotential: float):
    """Reads open circuit potential .mpt file from Biologic.

    Args:
        filename (str): Filename of .mpt file from Biologic OCV experiment.
        pH (float): pH of electrolyte for RHE conversion
        referencePotential (float): potential of reference electrode vs. SHE in V

    Returns:
        pd.DataFrame: Contains 'mode', 'error', 'Ewe/V', 'Ece/V', 'Analog IN 2/V', 'Ewe/mV'
    """
    #finds number of header lines
    with open(filename, 'r') as file:
        file.readline()
        line = file.readline()
        numHeaderLines = int(re.findall(r'-?\d*\.?\d+', line)[0])
        
    data = pd.read_csv(filename,
                        sep='\s+',
                        skiprows=numHeaderLines,
                        names = ['mode',
                                 'error',
                                 'time/s',
                                 'Ewe/V',
                                 'Ece/V',
                                 'Analog IN 2/V',
                                 'Ewe-Ece/V'],
                        index_col=False,
                        dtype = np.float64,
                        encoding='windows-1252')
    
    data['Ewe/V'] = data['Ewe/V'] + referencePotential + 0.059*pH
    data['Ece/V'] = data['Ece/V'] + referencePotential + 0.059*pH
    data['Ewe/mV'] = data['Ewe/V']*1000
    
    return data