import argparse
import decimal
decimal.getcontext().Emax = 1000000000
import math

def roundToNearestEvenInteger(number: float):
    """Rounds a float to the nearest even integer. Helper function for makeWaveform().

    Args:
        number (float): Any real number.

    Returns:
        int: Nearest even integer to number.
    """
    round1 = math.floor(number)
    round2 = math.ceil(number)
    
    if round1 % 2 == 0:
        return int(round1)
    else:
        return int(round2)

def makeWaveform(pulseWidth: decimal.Decimal, frequency: decimal.Decimal, faradaicBias: float, upBias: float, dnBias: float, oldWFG: bool):
    """Makes pulsing waveform that Siglent waveform generator can read.

    Args:
        pulseWidth (Decimal): Width of up and down pulses in seconds.
        frequency (Decimal): Frequency of waveform in Hertz.
        faradaicBias (float): Faradaic bias in waveform in V.
        upBias (float): Bias of up pulse in V.
        dnBias (float): Bias of down pulse in V.
        oldWFG (bool): If true, designs waveform for older Siglent, if false, designs waveform for newer Siglent.

    Returns:
        str: Confirmation of creation or failure reason.
    """
    if oldWFG:
        maxSamples = 16384
        maxRate = 150e6
        print("old")
    else:
        maxSamples = 8388608
        maxRate = 75e6
    
    #defines decimals because floats round off numbers and can't produce good gcd
    pulseWidthDecimal = pulseWidth
    frequencyDecimal = frequency
    pulseWidth = float(pulseWidth)
    frequency = float(frequency)
    
    #finds if pulseWidth is too small (leftmost point of graph)
    if pulseWidth < 1 / maxRate:
        return 'Pulse width is too small.'
    
    #finds if duty cycle is too high (right side of graph)
    if frequency > 1 / (2 * pulseWidth):
        return 'Decrease frequency or decrease pulse width.'
    
    #finds if duty cycle is too low (left side of graph)
    if frequency < 1 / (maxSamples * pulseWidth):
        return 'Increase pulse width or increase frequency.'
    
    if not oldWFG: #can change numberOfSamples only in new WFG
        #finds if number of samples needs to be decreased due to high frequency
        if frequency > maxRate / maxSamples:
            numberOfSamples = roundToNearestEvenInteger(maxRate/frequency)
        else:
            numberOfSamples = int(maxSamples)
            
        if numberOfSamples < 8:
            return 'Decrease frequency.'
        
        #finds numberOfSamples that maximizes fit to intended frequency
        if (pulseWidth * frequency * numberOfSamples) % 1 != 0: #not an integer, so will not give intended frequency
            
            numerator,denominator = (pulseWidthDecimal * frequencyDecimal).as_integer_ratio()
            
            if denominator < numberOfSamples:
                numberOfSamples -= (numberOfSamples % denominator)
                if numberOfSamples % 2 != 0: #ensures numberOfSamples is even
                    numberOfSamples -= denominator
                    
                    
            else: #if denominator is larger than numberOfSamples, cannot adjust maxSamples to frequency perfectly so this finds the number of samples that minimizes error
                
                minimizationVariable = 1 / (frequency * pulseWidth) #change this by integer multiples/divisors to minimize error
                integerIndex = 1
                
                if minimizationVariable >= numberOfSamples:
                    
                    while minimizationVariable >= numberOfSamples:
                        minimizationVariable /= integerIndex
                        integerIndex += 1
                        
                else:
                    
                    prevMinimizationVariable = minimizationVariable #need this because minimizationVariable MUST be smaller than numberOfSamples (can only decrease due to constraints set above)
                    
                    while minimizationVariable < numberOfSamples:
                        prevMinimizationVariable = minimizationVariable
                        minimizationVariable *= integerIndex
                        integerIndex += 1
                    
                    minimizationVariable = prevMinimizationVariable
                    
                numberOfSamples = roundToNearestEvenInteger(minimizationVariable)
    else:
        #checks if frequency is too high
        if frequency > maxRate / maxSamples:
            return 'Decrease frequency.'
        numberOfSamples = maxSamples
        
    upPulseEndPoint = round(numberOfSamples * pulseWidth * frequency)
    frequency = upPulseEndPoint / (pulseWidth * numberOfSamples)
    dnPulseStrtPoint = (numberOfSamples / 2) + 1
    dnPulseEndPoint = dnPulseStrtPoint + upPulseEndPoint
    upPulseEndPoint += 1
    
    #only numbers, capletters, lowerletters, underscores for valid filename
    filename = 'f_{frequency:E}_PW_{pulseWidth:1.3E}_F_{faradaicBias:+}_U_{upBias:+}_D_{dnBias:+}_{old}'.format(pulseWidth=pulseWidth,
                                                                                                        frequency=frequency,
                                                                                                        faradaicBias=faradaicBias,
                                                                                                        upBias=upBias,
                                                                                                        dnBias=dnBias,
                                                                                                        old=str(oldWFG))
    filename = filename.replace('-','n')
    filename = filename.replace('+','p')
    filename = filename.replace('.','d')
    filename = filename+'.csv'
    filename = 'C:\\Users\\tejas\\Analysis\\Potentiostat\\Waveforms\\'+filename
    
    amplitude = upBias - dnBias
    offset = (upBias + dnBias) / 2
    with open(filename,'w') as file:
        #writes header
        file.write('data length,{}\n'.format(numberOfSamples))
        file.write('frequency,{:.6f}\n'.format(frequency))
        file.write('amp,{:.6f}\n'.format(amplitude))
        file.write('offset,{:.6f}\n'.format(offset))
        file.write('phase,0.000000')
        file.write('\n\n\n\n\n\n\n\n')
        file.write('xpos,value\n')
        
        for i in range(1,numberOfSamples+1):
            #writes upBias
            if (i >= 1) and (i <= upPulseEndPoint):
                file.write('{},{:-.6f}\n'.format(i,upBias))
            #writes dnBias
            elif (i >= dnPulseStrtPoint) and (i <= dnPulseEndPoint):
                file.write('{},{:-.6f}\n'.format(i,dnBias))
            #writes faradaicBias
            else:
                file.write('{},{:-.6f}\n'.format(i,faradaicBias))
            
    return filename+' generated.'

def makePUND(pulseWidth: decimal.Decimal, frequency: decimal.Decimal, Vmid: float, Vmax: float, Vmin: float, oldWFG: bool):
    """Makes PUND pulsing waveform that Siglent waveform generator can read.

    Args:
        pulseWidth (Decimal): Width of up and down pulses in seconds.
        frequency (Decimal): Frequency of waveform in Hertz.
        Vmid (float): Mid-level bias in waveform in V.
        Vmax (float): Bias of positive pulses in V.
        Vmin (float): Bias of negative pulses in V.
        oldWFG (bool): If true, designs waveform for older Siglent, if false, designs waveform for newer Siglent.

    Returns:
        str: Confirmation of creation or failure reason.
    """
    if oldWFG:
        maxSamples = 16384
        maxRate = 150e6
        print("old")
    else:
        maxSamples = 8388608
        maxRate = 75e6
    
    #defines decimals because floats round off numbers and can't produce good gcd
    pulseWidthDecimal = pulseWidth
    frequencyDecimal = frequency
    pulseWidth = float(pulseWidth)
    frequency = float(frequency)
    
    #finds if pulseWidth is too small (leftmost point of graph)
    if pulseWidth < 1 / maxRate:
        return 'Pulse width is too small.'
    
    #finds if duty cycle is too high (right side of graph) - PUND has 4 pulses
    if frequency > 1 / (4 * pulseWidth):
        return 'Decrease frequency or decrease pulse width.'
    
    #finds if duty cycle is too low (left side of graph)
    if frequency < 1 / (maxSamples * pulseWidth):
        return 'Increase pulse width or increase frequency.'
    
    if not oldWFG: #can change numberOfSamples only in new WFG
        #finds if number of samples needs to be decreased due to high frequency
        if frequency > maxRate / maxSamples:
            numberOfSamples = roundToNearestEvenInteger(maxRate/frequency)
        else:
            numberOfSamples = int(maxSamples)
            
        if numberOfSamples < 8:
            return 'Decrease frequency.'
        
        #finds numberOfSamples that maximizes fit to intended frequency
        if (pulseWidth * frequency * numberOfSamples) % 1 != 0: #not an integer, so will not give intended frequency
            
            numerator,denominator = (pulseWidthDecimal * frequencyDecimal).as_integer_ratio()
            
            if denominator < numberOfSamples:
                numberOfSamples -= (numberOfSamples % denominator)
                if numberOfSamples % 2 != 0: #ensures numberOfSamples is even
                    numberOfSamples -= denominator
                    
                    
            else: #if denominator is larger than numberOfSamples, cannot adjust maxSamples to frequency perfectly so this finds the number of samples that minimizes error
                
                minimizationVariable = 1 / (frequency * pulseWidth) #change this by integer multiples/divisors to minimize error
                integerIndex = 1
                
                if minimizationVariable >= numberOfSamples:
                    
                    while minimizationVariable >= numberOfSamples:
                        minimizationVariable /= integerIndex
                        integerIndex += 1
                        
                else:
                    
                    prevMinimizationVariable = minimizationVariable #need this because minimizationVariable MUST be smaller than numberOfSamples (can only decrease due to constraints set above)
                    
                    while minimizationVariable < numberOfSamples:
                        prevMinimizationVariable = minimizationVariable
                        minimizationVariable *= integerIndex
                        integerIndex += 1
                    
                    minimizationVariable = prevMinimizationVariable
                    
                numberOfSamples = roundToNearestEvenInteger(minimizationVariable)
    else:
        #checks if frequency is too high
        if frequency > maxRate / maxSamples:
            return 'Decrease frequency.'
        numberOfSamples = maxSamples
    
    # Calculate pulse positions for PUND (4 pulses evenly spaced)
    samplesPerPulse = round(numberOfSamples * pulseWidth * frequency)
    frequency = samplesPerPulse / (pulseWidth * numberOfSamples)
    
    # Divide the period into 4 equal sections for even spacing
    sectionSize = numberOfSamples // 4
    
    # First positive pulse
    p1_start = 1
    p1_end = p1_start + samplesPerPulse
    
    # Second positive pulse 
    p2_start = sectionSize + 1
    p2_end = p2_start + samplesPerPulse
    
    # First negative pulse
    n1_start = 2 * sectionSize + 1
    n1_end = n1_start + samplesPerPulse
    
    # Second negative pulse
    n2_start = 3 * sectionSize + 1
    n2_end = n2_start + samplesPerPulse
    
    #only numbers, capletters, lowerletters, underscores for valid filename
    filename = 'PUND_f_{frequency:E}_PW_{pulseWidth:1.3E}_Mid_{Vmid:+}_Max_{Vmax:+}_Min_{Vmin:+}_{old}'.format(
                                                                                                        pulseWidth=pulseWidth,
                                                                                                        frequency=frequency,
                                                                                                        Vmid=Vmid,
                                                                                                        Vmax=Vmax,
                                                                                                        Vmin=Vmin,
                                                                                                        old=str(oldWFG))
    filename = filename.replace('-','n')
    filename = filename.replace('+','p')
    filename = filename.replace('.','d')
    filename = filename+'.csv'
    filename = 'C:\\Users\\tejas\\Analysis\\Potentiostat\\Waveforms\\'+filename
    
    amplitude = Vmax - Vmin
    offset = (Vmax + Vmin) / 2
    with open(filename,'w') as file:
        #writes header
        file.write('data length,{}\n'.format(numberOfSamples))
        file.write('frequency,{:.6f}\n'.format(frequency))
        file.write('amp,{:.6f}\n'.format(amplitude))
        file.write('offset,{:.6f}\n'.format(offset))
        file.write('phase,0.000000')
        file.write('\n\n\n\n\n\n\n\n')
        file.write('xpos,value\n')
        
        for i in range(1,numberOfSamples+1):
            #writes first positive pulse
            if (i >= p1_start) and (i <= p1_end):
                file.write('{},{:-.6f}\n'.format(i,Vmax))
            #writes second positive pulse
            elif (i >= p2_start) and (i <= p2_end):
                file.write('{},{:-.6f}\n'.format(i,Vmax))
            #writes first negative pulse
            elif (i >= n1_start) and (i <= n1_end):
                file.write('{},{:-.6f}\n'.format(i,Vmin))
            #writes second negative pulse
            elif (i >= n2_start) and (i <= n2_end):
                file.write('{},{:-.6f}\n'.format(i,Vmin))
            #writes Vmid (baseline)
            else:
                file.write('{},{:-.6f}\n'.format(i,Vmid))
            
    return filename+' generated.'
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', help='Frequency in Hz')
    parser.add_argument('--pw', help='Pulse width in seconds')
    parser.add_argument('--up', help='Up FE poling bias in volts')
    parser.add_argument('--dn', help='Down FE poling bias in volts')
    parser.add_argument('--faradaic', help='Faradaic bias in volts')
    parser.add_argument('--pund',help='If true, creates PUND waveform instead of H2 gen waveform.', default=False)
    args = parser.parse_args()
    if args.pund:
        print(makePUND(decimal.Decimal(args.pw),decimal.Decimal(args.freq),float(args.faradaic),float(args.up),float(args.dn),False))
    else:
        print(makeWaveform(decimal.Decimal(args.pw),decimal.Decimal(args.freq),float(args.faradaic),float(args.up),float(args.dn),False))