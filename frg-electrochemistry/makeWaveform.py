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

def makeWaveform(pulseWidth: float, frequency: float, faradaicBias: float, upBias: float, dnBias: float, oldWFG: bool):
    """Makes pulsing waveform that Siglent waveform generator can read.

    Args:
        pulseWidth (float): Width of up and down pulses in seconds.
        frequency (float): Frequency of waveform in Hertz.
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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq')
    parser.add_argument('--pw')
    parser.add_argument('--up')
    parser.add_argument('--dn')
    parser.add_argument('--faradaic')
    args = parser.parse_args()
    print(makeWaveform(decimal.Decimal(args.pw),decimal.Decimal(args.freq),float(args.faradaic),float(args.up),float(args.dn),False))