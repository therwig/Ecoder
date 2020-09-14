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
        #print ( kernel*kernel*inX*inY, -4*(2*kernel-1), -2*kernel*(inX+inY-2*kernel+2) )
        return inChan*nFilter*(kernel*kernel*inX*inY-4*(2*kernel-1)-2*kernel*(inX+inY-2*kernel+2))
    elif kernel==5:
        return inChan*nFilter*(kernel*kernel*inX*inY-8*(5*kernel-4)-4*kernel*(inX+inY-2*kernel+2))
    else: 
        print("unsupported kernel size!")
        return 0



#nominal model
inX=4
inY=4 
inChan=3
kernel=3 
nFilter=8
nOut=16

cnn = CNN_MACs(inX, inY, inChan, kernel, nFilter)
dense = (inX*inY*nFilter)*nOut

print("nominal: {}+{} = {}".format(cnn,dense,cnn+dense))

nOut=12
dense = (inX*inY*nFilter)*nOut
print("d12: {}+{} = {}".format(cnn,dense,cnn+dense))

nOut=20
dense = (inX*inY*nFilter)*nOut
print("d20: {}+{} = {}".format(cnn,dense,cnn+dense))

nOut=16
dense = (inX*inY*nFilter)*nOut + 16*16
print("den16: {}+{} = {}".format(cnn,dense,cnn+dense))

kernel=5
cnn = CNN_MACs(inX, inY, inChan, kernel, nFilter)
dense = (inX*inY*nFilter)*nOut
print("k5: {}+{} = {}".format(cnn,dense,cnn+dense))

kernel=3
nFilter=12
cnn = CNN_MACs(inX, inY, inChan, kernel, nFilter)
dense = (inX*inY*nFilter)*nOut
print("c12: {}+{} = {}".format(cnn,dense,cnn+dense))

nFilter=8
cnn = CNN_MACs(inX, inY, inChan, kernel, nFilter)
dense = (inX*inY*nFilter)*nOut / 4 #2x2 pool factor
print("pool: {}+{} = {}".format(cnn,dense,cnn+dense))

#multilayer setups
nF1=8
nF2=8
cnn1 = CNN_MACs(inX, inY, inChan, kernel, nF1)
cnn2 = CNN_MACs(inX, inY, nF1, kernel, nF2)
dense = (inX*inY*nF2)*nOut
print("c8,8: {}+{} = {}".format(cnn1+cnn2,dense,cnn1+cnn2+dense))

nF1=8
nF2=8
nF3=8
cnn1 = CNN_MACs(inX, inY, inChan, kernel, nF1)
cnn2 = CNN_MACs(inX, inY, nF1, kernel, nF2)
cnn3 = CNN_MACs(inX, inY, nF2, kernel, nF3)
dense = (inX*inY*nF3)*nOut
print("c8,8,8: {}+{} = {}".format(cnn1+cnn2+cnn3,dense,cnn1+cnn2+cnn3+dense))

nF1=4
nF2=4
nF3=4
cnn1 = CNN_MACs(inX, inY, inChan, kernel, nF1)
cnn2 = CNN_MACs(inX, inY, nF1, kernel, nF2)
cnn3 = CNN_MACs(inX, inY, nF2, kernel, nF3)
dense = (inX*inY*nF3)*nOut
print("c4,4,4: {}+{} = {}".format(cnn1+cnn2+cnn3,dense,cnn1+cnn2+cnn3+dense))

nF1=8
nF2=4
nF3=2
cnn1 = CNN_MACs(inX, inY, inChan, kernel, nF1)
cnn2 = CNN_MACs(inX, inY, nF1, kernel, nF2)
cnn3 = CNN_MACs(inX, inY, nF2, kernel, nF3)
dense = (inX*inY*nF3)*nOut
print("c8,4,2: {}+{} = {}".format(cnn1+cnn2+cnn3,dense,cnn1+cnn2+cnn3+dense))

nF1=8
nF2=4
nF3=4
nF4=4
nF5=2
cnn1 = CNN_MACs(inX, inY, inChan, kernel, nF1)
cnn2 = CNN_MACs(inX, inY, nF1, kernel, nF2)
cnn3 = CNN_MACs(inX, inY, nF2, kernel, nF3)
cnn4 = CNN_MACs(inX, inY, nF3, kernel, nF4)
cnn5 = CNN_MACs(inX, inY, nF4, kernel, nF5)
dense = (inX*inY*nF3)*nOut
print("c8,4,4,4,2: {}+{} = {}".format(cnn1+cnn2+cnn3+cnn4+cnn5,
                                      dense,
                                      cnn1+cnn2+cnn3+cnn4+cnn5+dense))

nF1=8
nF2=4
nF3=2
cnn1 = CNN_MACs(inX, inY, inChan, 5, nF1)
cnn2 = CNN_MACs(inX, inY, nF1, 5, nF2)
cnn3 = CNN_MACs(inX, inY, nF2, 3, nF3)
dense = (inX*inY*nF3)*nOut
print("c8,4,2 k5,5,3: {}+{} = {}".format(cnn1+cnn2+cnn3,dense,cnn1+cnn2+cnn3+dense))

nF1=8
nF2=4
nF3=4
cnn1 = CNN_MACs(inX, inY, inChan, 5, nF1)
cnn2 = CNN_MACs(inX, inY, nF1, 5, nF2)
cnn3 = CNN_MACs(inX, inY, nF2, 3, nF3)
dense = (inX*inY*nF3)*nOut
print("c8,4,4 k5,5,3: {}+{} = {}".format(cnn1+cnn2+cnn3,dense,cnn1+cnn2+cnn3+dense))

nF1=4
nF2=4
nF3=4
cnn1 = CNN_MACs(inX, inY, inChan, 5, nF1)
cnn2 = CNN_MACs(inX, inY, nF1, 5, nF2)
cnn3 = CNN_MACs(inX, inY, nF2, 3, nF3)
dense = (inX*inY*nF3)*nOut + 16*16
print("c4,4,4 k5,5,3: {}+{} = {}".format(cnn1+cnn2+cnn3,dense,cnn1+cnn2+cnn3+dense))
