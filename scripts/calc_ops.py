#test


def CNN_MACs(inX, inY, inChan, kernel, nFilter, pad=True):
    '''
    number of MACs per CNN inference.
    assumes kernel size is smaller than input image
    C. Herwig 2020
    '''
    if not pad:
        return inChan*nFilter*(kernel*kernel)*(inX-kernel+1)*(inY-kernel+1)
    elif kernel==3:
        return inChan*nFilter*(kernel*kernel)*(inX*inY-4*(2*kernel-1)-2*(inX+inY-2*kernel))
    elif kernel==5:
        return inChan*nFilter*(kernel*kernel)*(inX*inY-8*(5*kernel-4)-4*(inX+inY-2*kernel))
    else: 
        print("unsupported kernel size!")
        return 0



