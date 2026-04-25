import pandas as pd
import numpy as np
import matplotlib as mpl
import scipy as sc
import re
import os
import datetime

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_mpt_with_headers(filename: str) -> pd.DataFrame:
    """Reads a Biologic .mpt file with auto-detected header count and column names.

    The second line of every .mpt file contains 'Nb header lines : N'; the Nth
    line is the tab-separated column header row.

    Args:
        filename (str): path to .mpt file

    Returns:
        pd.DataFrame: raw data with Biologic column names
    """
    with open(filename, 'r', encoding='windows-1252') as file:
        file.readline()
        line = file.readline()
        numHeaderLines = int(re.findall(r'-?\d*\.?\d+', line)[0])

        headers = []
        file.seek(0)
        currLine = 1
        for line in file:
            if currLine == numHeaderLines:
                headers = line.strip().split('\t')
            currLine += 1

    return pd.read_csv(filename,
                       sep=r'\s+',
                       skiprows=numHeaderLines,
                       names=headers,
                       index_col=False,
                       dtype=np.float64,
                       encoding='windows-1252')


def _alias_biologic_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalizes Biologic column aliases in place: averaged columns like
    <I>/mA, <Ewe>/V, <Ewe/V> are mapped to the canonical I/mA, Ewe/V.
    """
    if '<I>/mA' in data.columns:
        data['I/mA'] = data['<I>/mA']
    if '<Ewe>/V' in data.columns:
        data['Ewe/V'] = data['<Ewe>/V']
    if '<Ewe/V>' in data.columns:
        data['Ewe/V'] = data['<Ewe/V>']
    return data


def _default_rs_from_pH(pH: float, verbose: bool = True) -> float:
    """pH-based Rs fallback used when Rcmp/Ohm is unavailable."""
    if pH == 13:
        rs = 45
    else:  # pH 14 usually
        rs = 5
    if verbose:
        print(f'could not find Rs, defaulting to {rs}')
    return rs


def _resolve_experimental_rs(data: pd.DataFrame, pH: float, verbose: bool = True) -> float:
    """Get the Rs value the potentiostat actually used — from Rcmp/Ohm if present,
    otherwise pH-based default.
    """
    if 'Rcmp/Ohm' in data.columns:
        return data['Rcmp/Ohm'].mean()
    return _default_rs_from_pH(pH, verbose=verbose)


def _apply_ir_and_rhe_corrections(data: pd.DataFrame,
                                  pH: float,
                                  area: float,
                                  referencePotential: float,
                                  compensationAmount: float,
                                  solutionResistance: float = None,
                                  assumedExperimentalComp: float = 0.85,
                                  addIAColumn: bool = False) -> pd.DataFrame:
    """Applies the standard Biologic post-processing pipeline shared by CP, CV, CA.

    Pipeline:
      1. Alias <I>/mA, <Ewe>/V, <Ewe/V> to canonical names.
      2. iR compensation:
         - If solutionResistance is None: use the experimental Rs (Rcmp/Ohm mean
           or pH default) and subtract compensationAmount * Rs * I.
         - If solutionResistance is given: first undo the experimental correction
           the potentiostat applied (assumedExperimentalComp * Rs_experimental * I),
           then subtract compensationAmount * solutionResistance * I. This matches
           the readCP override pattern.
      3. RHE shift (Ewe/V += referencePotential + 0.059*pH), current density,
         Ewe/mV, j/A*m-2, optional I/A.
      4. Drop Synology zero-artifact rows (time/s == 0 and I/mA == 0).

    Args:
        data (pd.DataFrame): raw Biologic data (should contain time/s, I/mA, Ewe/V).
        pH (float): pH of electrolyte.
        area (float): electrode area in cm^2.
        referencePotential (float): reference potential vs. SHE in V.
        compensationAmount (float): fraction of iR to subtract in software
            (0.15 by default in the original code — i.e. the 15% the potentiostat
            did not compensate in hardware).
        solutionResistance (float, optional): override Rs in Ohms. When given,
            the experimental compensation is first undone before applying this.
        assumedExperimentalComp (float, default 0.85): the hardware compensation
            fraction the experiment was run with. Only used when solutionResistance
            is supplied.
        addIAColumn (bool, default False): if True, adds I/A column (readCV does this).

    Returns:
        pd.DataFrame: corrected data (same object, modified).
    """
    _alias_biologic_columns(data)

    if solutionResistance is None:
        rs = _resolve_experimental_rs(data, pH)
        data['Ewe/V'] = data['Ewe/V'] - (data['I/mA'] * 0.001 * rs * compensationAmount)
    else:
        rs_experimental = _resolve_experimental_rs(data, pH)
        # undo what the potentiostat did in hardware
        data['Ewe/V'] = data['Ewe/V'] + (data['I/mA'] * 0.001 * rs_experimental * assumedExperimentalComp)
        # apply custom correction
        data['Ewe/V'] = data['Ewe/V'] - (data['I/mA'] * 0.001 * solutionResistance * compensationAmount)

    if addIAColumn:
        data['I/A'] = data['I/mA'] / 1000

    data['j/mA*cm-2'] = data['I/mA'] / area
    data['Ewe/V'] = data['Ewe/V'] + referencePotential + 0.059 * pH
    data['Ewe/mV'] = data['Ewe/V'] * 1000
    data['j/A*m-2'] = data['j/mA*cm-2'] * 10

    # cleans data to remove points where Synology interferes with saving data
    data = data[~((data['time/s'] == 0) & (data['I/mA'] == 0))]

    return data


def _build_list(reader, filenameList, *args, **kwargs):
    """Generic list builder: applies a reader function across a list of filenames."""
    return [reader(f, *args, **kwargs) for f in filenameList]


# ---------------------------------------------------------------------------
# Biologic readers
# ---------------------------------------------------------------------------

def readCP(filename: str, pH: float, area: float, referencePotential: float,
           compensationAmount: float = 0.15, solutionResistance: float = None):
    """Reads chronopotentiometry data from Biologic into a Pandas dataframe.

    Args:
        filename (str): filename of CP .mpt from Biologic
        pH (float): pH of electrolyte for RHE conversion
        area (float): geometric area of electrode in cm^2 for current density
        referencePotential (float): potential of reference electrode vs. SHE in V
        compensationAmount (float, default 0.15): amount of iR to subtract in software
        solutionResistance (float, default None): custom Rs in Ohms; when set,
            undoes the experimental 85% hardware compensation first.

    Returns:
        pd.DataFrame: dataframe of CP data
    """
    data = _read_mpt_with_headers(filename)
    return _apply_ir_and_rhe_corrections(data, pH, area, referencePotential,
                                         compensationAmount, solutionResistance)


def buildCPList(filenameList: list[str], pH: float, area: float, referencePotential: float,
                compensationAmount: float = 0.15, solutionResistance: float = None):
    """Takes list of filenames with same pH, area, and referencePotential and builds CP list.

    Args:
        filenameList (list[str]): filenames to read
        pH, area, referencePotential: see readCP
        compensationAmount, solutionResistance: see readCP

    Returns:
        list[pd.DataFrame]
    """
    return _build_list(readCP, filenameList, pH, area, referencePotential,
                       compensationAmount, solutionResistance)


def readCV(filename: str, pH: float, area: float, referencePotential: float,
           compensationAmount: float = 0.15, solutionResistance: float = None):
    """Reads cyclic voltammetry data from Biologic into a Pandas dataframe.

    Args:
        filename (str): .mpt file from Biologic, or legacy "CV-all" .txt export
        pH (float): pH of electrolyte for RHE conversion
        area (float): geometric area of electrode in cm^2 for current density
        referencePotential (float): potential of reference electrode vs. SHE in V
        compensationAmount (float, default 0.15): amount of iR to subtract in software
        solutionResistance (float, default None): custom Rs in Ohms; when set,
            undoes the experimental 85% hardware compensation first.

    Returns:
        pd.DataFrame: dataframe of CV data
    """
    if filename[-3:] == 'mpt':
        data = _read_mpt_with_headers(filename)
    else:
        # legacy CV-all .txt export — hardcoded column order
        data = pd.read_csv(filename,
                           sep=r'\s+',
                           skiprows=1,
                           names=['mode', 'ox/red', 'error', 'control changes',
                                  'counter inc.', 'time/s', 'control/V',
                                  'Ewe/V', 'I/mA', 'cycle number', '(Q-Qo)/C',
                                  'I Range', '<Ece>/V', 'Analog IN 2/V',
                                  'Rcmp/Ohm', 'P/W', 'Ewe-Ece/V'],
                           index_col=False,
                           dtype=np.float64,
                           engine='python')

    return _apply_ir_and_rhe_corrections(data, pH, area, referencePotential,
                                         compensationAmount, solutionResistance,
                                         addIAColumn=True)


def buildCVList(filenameList: list[str], pH: float, area: float, referencePotential: float,
                compensationAmount: float = 0.15, solutionResistance: float = None):
    """Takes list of filenames with same pH, area, and referencePotential and builds CV list.

    Args:
        filenameList (list[str]): filenames to read
        pH, area, referencePotential: see readCV
        compensationAmount, solutionResistance: see readCV

    Returns:
        list[pd.DataFrame]
    """
    return _build_list(readCV, filenameList, pH, area, referencePotential,
                       compensationAmount, solutionResistance)


def buildEDLCList(folderName: str, number: int, pH: float, area: float, referencePotential: float,
                  excludeLastX: int = 0,
                  compensationAmount: float = 0.15, solutionResistance: float = None):
    """Wrapper around buildCVList for faster EDLC analysis.

    Args:
        folderName (str): Folder that EDLC CVs are located in.
        number (int): Number that EDLC CVs all start with in common. Matches any
            leading-zero variant (e.g. number=5 matches '5_', '05_', '005_').
        pH (float): pH that EDLC was taken at.
        area (float): Area of electrode for EDLC in cm^2.
        referencePotential (float): Potential of reference electrode
        excludeLastX (int, optional): Excludes last x files. Defaults to 0.
        compensationAmount, solutionResistance: forwarded to readCV.

    Returns:
        list[pd.DataFrame]: List of CVs for EDLC
    """
    if not os.path.isdir(folderName):
        return None

    files = os.listdir(folderName)
    edlcFiles = []
    for file in files:
        m = re.match(r'^(\d+)_', file)
        isEDLC = m is not None and int(m.group(1)) == number
        if (file[-3:] != 'txt') and (file[-3:] != 'mpt'):
            isEDLC = False
        if ('CA' in file) or ('WAIT' in file) or ('OCV' in file):
            isEDLC = False
        if isEDLC:
            edlcFiles.append(folderName + '\\' + file)

    if excludeLastX != 0:
        edlcFiles = edlcFiles[excludeLastX:]

    base_mpt = {f[:-4] for f in edlcFiles if f.endswith('.mpt')}
    cleaned_list = [f for f in edlcFiles if not (f.endswith('.txt') and f[:-4] in base_mpt)]

    return buildCVList(cleaned_list, pH, area, referencePotential,
                       compensationAmount=compensationAmount,
                       solutionResistance=solutionResistance)


def readCA(filename: str, pH: float, area: float, referencePotential: float,
           shouldRemoveNoise: bool = False,
           compensationAmount: float = 0.15, solutionResistance: float = None):
    """Reads chronoamperometry data from Biologic into a Pandas dataframe.

    Args:
        filename (str): .mpt file from Biologic, or legacy .txt export
        pH (float): pH of electrolyte for RHE conversion
        area (float): geometric area of electrode in cm^2 for current density
        referencePotential (float): potential of reference electrode vs. SHE in V
        shouldRemoveNoise (bool, default False): apply notch filter via removeNoise.
        compensationAmount (float, default 0.15): amount of iR to subtract in software
        solutionResistance (float, default None): custom Rs in Ohms; when set,
            undoes the experimental 85% hardware compensation first.

    Returns:
        pd.DataFrame: dataframe of CA data
    """
    if filename[-3:] == 'mpt':
        data = _read_mpt_with_headers(filename)
    else:
        # legacy CA .txt export — hardcoded column order
        data = pd.read_csv(filename,
                           sep=r'\s+',
                           skiprows=1,
                           names=['mode', 'ox/red', 'error', 'control changes',
                                  'Ns changes', 'counter inc.', 'Ns', 'time/s',
                                  'control/V', 'Ewe/V', 'I/mA', 'dQ/C',
                                  'I range', 'Ece/V', 'Analog IN 2/V',
                                  'Rcmp/Ohm', 'Capacitance charge/muF',
                                  'Capacitance discharge/muF', 'Efficiency/%',
                                  'cycle number', 'P/W'],
                           index_col=False,
                           dtype=np.float64,
                           encoding='unicode_escape')

    # alias columns first so removeNoise can find I/mA
    _alias_biologic_columns(data)

    if shouldRemoveNoise:
        data = removeNoise(data)

    return _apply_ir_and_rhe_corrections(data, pH, area, referencePotential,
                                         compensationAmount, solutionResistance)


def buildCAList(filenameList: list[str], pH: float, area: float, referencePotential: float,
                shouldRemoveNoise: bool = False,
                compensationAmount: float = 0.15, solutionResistance: float = None):
    """Takes list of filenames with same pH, area, and referencePotential and builds CA list.

    Args:
        filenameList (list[str]): filenames to read
        pH, area, referencePotential: see readCA
        shouldRemoveNoise, compensationAmount, solutionResistance: see readCA

    Returns:
        list[pd.DataFrame]
    """
    return _build_list(readCA, filenameList, pH, area, referencePotential,
                       shouldRemoveNoise, compensationAmount, solutionResistance)


def readPEIS(filename: str, freqRange: list[float] = None):
    """Reads potentiostatic electrochemical impedance spectroscopy data from Biologic.

    Args:
        filename (str): .mpt file, or legacy "PEIS-all" .txt export
        freqRange (list[float]): [low, high] frequency range to keep. Defaults to None.

    Returns:
        pd.DataFrame: dataframe of PEIS data
    """
    if filename[-3:] == 'mpt':
        data = _read_mpt_with_headers(filename)
    else:
        # legacy PEIS-all .txt export
        data = pd.read_csv(filename,
                           sep=r'\s+',
                           skiprows=1,
                           names=['freq/Hz', 'Re(Z)/Ohm', '-Im(Z)/Ohm',
                                  '|Z|/Ohm', 'Phase(Z)/deg', 'time/s',
                                  '<Ewe>/V', '<I>/mA', 'Cs/uF', 'Cp/uF',
                                  'cycle number', 'I Range', '|Ewe|/V',
                                  '|I|/A', 'Ns', '(Q-Qo)/mA.h',
                                  'Re(Y)/Ohm-1', 'Im(Y)/Ohm-1', '|Y|/Ohm-1',
                                  'Phase(Y)/deg', 'dq/mA.h'],
                           index_col=False,
                           dtype=np.float64,
                           encoding='unicode_escape')

    # clean Synology zero-artifact rows
    data = data[~((data['time/s'] == 0) & (data['<I>/mA'] == 0))]

    if freqRange is not None:
        data = data[(data['freq/Hz'] >= freqRange[0]) & (data['freq/Hz'] <= freqRange[1])]

    data.attrs['filename'] = filename
    return data


def buildPEISList(filenameList: list[str], freqRange: list[float] = None):
    """Takes list of filenames and builds EIS list.

    Args:
        filenameList (list[str]): filenames to read
        freqRange (list[float]): see readPEIS

    Returns:
        list[pd.DataFrame]
    """
    return _build_list(readPEIS, filenameList, freqRange)


def readPEISImpedance(filename: str, freqRange: list[float] = None):
    """Reads PEIS directly into the format impedance.py / bayesdrt2 expects.

    Args:
        filename (str): filename of PEIS .mpt or legacy .txt
        freqRange (list[float]): see readPEIS

    Returns:
        tuple(np.ndarray(float), np.ndarray(complex)): (frequency, complex impedance)
    """
    return convertToImpedanceAnalysis(readPEIS(filename, freqRange))


def convertToImpedanceAnalysis(data: pd.DataFrame):
    """Converts dataframe to format that impedance.py or bayesdrt2 can use.

    Args:
        data (pd.DataFrame): Output of ReadDataFiles.readPEIS.

    Returns:
        tuple(np.ndarray(float), np.ndarray(complex)): (frequency, complex impedance)
    """
    frequency = data['freq/Hz'].to_numpy()
    realImpedance = data['Re(Z)/Ohm'].to_numpy()
    imagImpedance = -data['-Im(Z)/Ohm'].to_numpy()
    impedance = realImpedance + 1j * imagImpedance
    return (frequency, impedance.astype(np.complex128))


def readOSC(filename: str, pH: float, area: float, referencePotential: float, irange: str,
            stretch: float = 1, solutionResistance: float = 0, compensationAmount: float = 1):
    """Reads oscilloscope .csv from Picoscope.

    Args:
        filename (str): filename of waveform .csv from Picoscope
        pH (float): pH of electrolyte for RHE conversion
        area (float): geometric area of electrode in cm^2 for current density
        referencePotential (float): potential of reference electrode vs. SHE in V
        irange (str): '2A', '1A', '100mA', '10mA', '1mA', '100uA', '10uA'
        stretch (float, optional): For 1 Hz typically 4, for 10 Hz typically 2. Defaults to 1.
        solutionResistance (float, default 0): Rs to compensate for, in Ohms.
        compensationAmount (float, default 1): amount of iR to subtract.

    Returns:
        pd.DataFrame: dataframe containing oscilloscope values
    """
    with open(filename, 'r') as file:
        line1 = next(file)
        line2 = next(file)
        line3 = next(file)
        firstChannel = line1[33]
        unitsList = line2.split(',')
        unitsList = [i.replace('(', '').replace(')', '') for i in unitsList]
        timeUnit = unitsList[0]
        voltage1Unit = unitsList[3]
        voltage2Unit = unitsList[4]

    names = ['Time (xs)', 'discard1', 'discard2', 'ch1Avg', 'ch2Avg']

    data = pd.read_csv(filename,
                       skiprows=3,
                       engine='pyarrow',
                       sep=',',
                       names=names,
                       na_values=['∞', '-∞'],
                       dtype=np.float32,
                       on_bad_lines='skip')

    data = data.ffill()

    if 'mV' in voltage1Unit:
        data['ch1Avg'] = data['ch1Avg'] / 1000
    if 'mV' in voltage2Unit:
        data['ch2Avg'] = data['ch2Avg'] / 1000

    data['Time (xs)'] = data['Time (xs)'] - data['Time (xs)'].loc[0]

    if 'ms' in timeUnit:
        data['Time (ms)'] = data['Time (xs)']
    elif 'us' in timeUnit:
        data['Time (ms)'] = data['Time (xs)'] / 1000

    if 'A' in firstChannel:
        data['Voltage (V)'] = data['ch1Avg']
        data['Current (A)'] = data['ch2Avg']
    else:
        data['Voltage (V)'] = data['ch2Avg']
        data['Current (A)'] = data['ch1Avg']

    # estimates the picoscope range used for current to subtract offsets
    max_current_voltage = data['Current (A)'].abs().max()
    if max_current_voltage >= 5:
        data['Current (A)'] -= 14.1 / 1000
    elif max_current_voltage >= 2:
        data['Current (A)'] -= 4.58 / 1000
    elif max_current_voltage >= 1:
        data['Current (A)'] -= 4.43 / 1000
    elif max_current_voltage >= 0.5:
        data['Current (A)'] -= 2.73 / 1000
    elif max_current_voltage >= 0.2:
        data['Current (A)'] -= 1.95 / 1000
    else:
        print('WARNING: Likely used small Picoscope range.')

    if irange == '1A' or irange == '2A':
        data['Current (A)'] = data['Current (A)']
    elif irange == '100mA':
        data['Current (A)'] = data['Current (A)'] * 0.1
    elif irange == '10mA':
        data['Current (A)'] = data['Current (A)'] * 0.01
    elif irange == '1mA':
        data['Current (A)'] = data['Current (A)'] * 0.001
    elif (irange == '100uA') or (irange == '100µA'):
        data['Current (A)'] = data['Current (A)'] * 0.0001
    elif (irange == '10uA') or (irange == '10µA'):
        data['Current (A)'] = data['Current (A)'] * 0.00001

    data['Voltage (V)'] = data['Voltage (V)'] - (data['Current (A)'] * solutionResistance * compensationAmount)

    data['Time (ms)'] *= stretch
    data = data[data['Time (ms)'] < (data['Time (ms)'].max() / stretch)]

    data['Time (s)'] = data['Time (ms)'] / 1000

    data['RawVoltage (V)'] = data['Voltage (V)']
    data['Voltage (V)'] = data['Voltage (V)'] + referencePotential + 0.059 * pH

    chargeArray = sc.integrate.cumulative_trapezoid(data['Current (A)'], x=data['Time (s)'])
    chargeArray = np.insert(chargeArray, 0, 0)
    data['Charge (C)'] = pd.Series(chargeArray)
    data['Charge Density (mC/cm^2)'] = data['Charge (C)'] * 1000 / area

    data['Current (mA)'] = data['Current (A)'] * 1000
    data['Current Density (mA/cm^2)'] = data['Current (mA)'] / area

    data = data.drop(['discard1', 'discard2', 'Time (xs)', 'ch1Avg', 'ch2Avg'], axis=1)
    return data


def readRawWaveform(filename: str, pH: float, referencePotential: float):
    """Reads waveforms that Siglent can read into a Pandas DataFrame.

    Args:
        filename (str): filename of waveform .csv from makeWaveform.py or Siglent software.
        pH (float): pH of electrolyte for RHE conversion
        referencePotential (float): potential of reference electrode vs. SHE in V

    Returns:
        pd.DataFrame: with 'Time (s)', 'Time (ms)', 'RawVoltage (V)', 'Voltage (V)'
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
        dataLength = float(lines[0][12:])
        frequency = float(lines[1][10:])

    data = pd.read_csv(filename,
                       skiprows=12,
                       sep=',',
                       dtype=float)
    data['Time (s)'] = (data['xpos'] - 1) / (frequency * dataLength)
    data['Time (ms)'] = data['Time (s)'] * 1000
    data['RawVoltage (V)'] = data['value']
    data['Voltage (V)'] = data['value'] + referencePotential + 0.059 * pH
    return data


def readDRT(filename: str):
    """Reads DRT generated by pyDRTTools.

    Args:
        filename (str): filename of DRT .csv

    Returns:
        pd.DataFrame: dataframe with 'tau', 'gamma'
    """
    return pd.read_csv(filename, skiprows=2, sep=',', dtype=float)


def readOCV(filename: str, pH: float, referencePotential: float):
    """Reads open circuit potential .mpt file from Biologic.

    Args:
        filename (str): .mpt file from Biologic OCV experiment.
        pH (float): pH of electrolyte for RHE conversion
        referencePotential (float): potential of reference electrode vs. SHE in V

    Returns:
        pd.DataFrame: Contains 'mode', 'error', 'time/s', 'Ewe/V', 'Ece/V',
                      'Analog IN 2/V', 'Ewe-Ece/V', 'Ewe/mV'
    """
    # OCV files have a fixed column layout regardless of .mpt header count
    with open(filename, 'r', encoding='windows-1252') as file:
        file.readline()
        line = file.readline()
        numHeaderLines = int(re.findall(r'-?\d*\.?\d+', line)[0])

    data = pd.read_csv(filename,
                       sep=r'\s+',
                       skiprows=numHeaderLines,
                       names=['mode', 'error', 'time/s', 'Ewe/V', 'Ece/V',
                              'Analog IN 2/V', 'Ewe-Ece/V'],
                       index_col=False,
                       dtype=np.float64,
                       encoding='windows-1252')

    # clean Synology zero-artifact rows
    data = data[~((data['time/s'] == 0) & (data['Ewe/V'] == 0))]

    data['Ewe/V'] = data['Ewe/V'] + referencePotential + 0.059 * pH
    data['Ece/V'] = data['Ece/V'] + referencePotential + 0.059 * pH
    data['Ewe/mV'] = data['Ewe/V'] * 1000
    return data


# ---------------------------------------------------------------------------
# Utilities (unchanged)
# ---------------------------------------------------------------------------

def colorFader(c1: str, c2: str, currentIndex: int, totalIndices: int):
    """Generates color gradient for plotting.

    Args:
        c1 (str): color of first index
        c2 (str): color of last index
        currentIndex (int): index to linearly interpolate between colors
        totalIndices (int): total number of indices

    Returns:
        color: matplotlib color in hex format
    """
    if totalIndices > 1:
        mix = currentIndex / (totalIndices - 1)
    else:
        mix = 0
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def removeNoise(df: pd.DataFrame, fundamental_freq: float = 0.46, Q: float = 1):
    """
    Apply multiple notch filters to remove a fundamental frequency and its harmonics
    from an electrochemical signal.
    """
    filtered_df = df.copy()
    time = df['time/s'].values
    current = df['I/mA'].values

    time_diffs = np.diff(time)
    dt = np.median(time_diffs)
    fs = 1 / dt

    filtered_current = current.copy()

    belowNyquistFreq = True
    i = 1
    while belowNyquistFreq:
        target_freq = i * fundamental_freq
        if target_freq > fs / 2:
            belowNyquistFreq = False
            break
        b, a = sc.signal.iirnotch(target_freq, Q, fs)
        filtered_current = sc.signal.filtfilt(b, a, filtered_current)
        i += 1

    filtered_df['I/mA'] = filtered_current
    return filtered_df


def calculateIntegral(timeSeries: pd.Series, valueSeries: pd.Series, baseline: float, timeBounds: list[float]):
    """Calculates 1D integral of (valueSeries - baseline) over timeSeries, bounded by timeBounds."""
    timeSeries = timeSeries[(timeSeries >= timeBounds[0]) & (timeSeries <= timeBounds[1])]
    valueSeries = valueSeries.loc[timeSeries.first_valid_index():timeSeries.last_valid_index()]
    valueSeries = valueSeries - baseline
    return sc.integrate.trapezoid(y=valueSeries, x=timeSeries)


def readExcelSheet(filename: str, area: float):
    """Gets H2 produced from Excel sheet.

    Args:
        filename (str): .xlsx file with GC data
        area (float): area of electrode in cm^2 for normalization

    Returns:
        dict[experimentNumber] = (H2 Value (mol/cm^2), H2 Error (mol/cm^2))
    """
    finalDict = {}
    xl = pd.ExcelFile(filename)

    for sheet_name in xl.sheet_names:
        try:
            experimentNumber = int(re.search(r'\d+', str(sheet_name)).group())
        except Exception:
            continue

        df = pd.read_excel(filename, sheet_name=sheet_name, header=None)

        foundValue = False
        h2prod = h2err = None
        for row_idx, row in df.iterrows():
            for col_idx, value in enumerate(row):
                if isinstance(value, str) and "Total H2 Generated (mol)" in value:
                    h2prod = np.float64(df.iloc[row_idx, col_idx + 1])
                    h2err = np.float64(df.iloc[row_idx, col_idx + 2])
                    foundValue = True
                    break
            if foundValue:
                break

        if foundValue:
            finalDict[experimentNumber] = (h2prod / area, h2err / area)

    return finalDict


def buildTechniqueList(folder_path: str, techniqueName: str, printOut: bool = True):
    """
    Finds all files of a specific technique from .mpt files in a folder
    and returns them in chronological order (earliest acquisition first).

    Args:
        folder_path (str): Path to the folder to search in.
        techniqueName (str): 'Cyclic Voltammetry',
                             'Potentio Electrochemical Impedance Spectroscopy',
                             'Chronoamperometry / Chronocoulometry',
                             'Dynamic Chronoamperometry'
        printOut (bool): print list of filenames

    Returns:
        list[str]: sorted list of full file paths
    """
    matching_files = []
    search_string = techniqueName

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' does not exist.")
        return matching_files

    if techniqueName != 'Dynamic Chronoamperometry':
        mpt_files = [f for f in os.listdir(folder_path) if f.endswith('.mpt')]

        if not mpt_files:
            print(f"No .mpt files found in '{folder_path}'.")
            return matching_files

        file_info = []
        for filename in mpt_files:
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='windows-1252') as file:
                    lines = file.readlines(1000)
                    if len(lines) >= 4:
                        fourth_line = lines[3]
                        secondLine = lines[1]
                        if search_string in fourth_line:
                            acquisition_time = None
                            for line in lines:
                                if "Acquisition started on :" in line:
                                    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2}\.\d{3})', line)
                                    if date_match:
                                        date_str = date_match.group(1)
                                        try:
                                            acquisition_time = datetime.datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S.%f')
                                            break
                                        except ValueError:
                                            try:
                                                acquisition_time = datetime.datetime.strptime(date_str, '%d/%m/%Y %H:%M:%S.%f')
                                                break
                                            except ValueError:
                                                pass

                            if acquisition_time:
                                file_info.append((file_path, acquisition_time))
                            else:
                                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                                file_info.append((file_path, mod_time))
                        elif 'Nb header lines : 3' in secondLine:
                            if search_string == 'Chronoamperometry / Chronocoulometry':
                                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                                file_info.append((file_path, mod_time))
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    else:
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        file_info = []
        for filename in csv_files:
            m = re.match(r"(\d+)_", filename)
            file_path = os.path.join(folder_path, filename)
            file_info.append((file_path, int(m.group(1))))

    file_info.sort(key=lambda x: x[1])
    matching_files = [info[0] for info in file_info]

    if printOut:
        for i, file in enumerate(matching_files):
            print(str(i) + ' ' + file)
            print('\n')

    return matching_files


def split_by_known_prefixes(file_paths, known_prefixes):
    """Split a list of file paths based on known filename prefixes."""
    result = {prefix: [] for prefix in known_prefixes}
    result["unmatched"] = []

    for path in file_paths:
        filename = os.path.basename(path)
        matched = False
        for prefix in known_prefixes:
            if filename.startswith(prefix):
                result[prefix].append(path)
                matched = True
                break
        if not matched:
            result["unmatched"].append(path)

    return result