import numpy as np
from dtcwt_matlab import FSfarras

# Example 1 (Test de comparación con Matlab)
#x = np.array([1,2,3,4,5,6])
#Faf, Fsf = FSfarras()
#af, sf = dualfilt1()
#lo, hi = afb(x, af[0])
#y = sfb(lo,hi,sf[0])
#print('done')

# Example 2
#x = np.ones(512)            # Test signal
#J = 4                       # Number of stages
#Faf, Fsf = FSfarras()       # 1st stage analysis and synthesis filters
#af, sf = dualfilt1()        # Remaining stages anal. and synth. filters
#z, w = dualtree1D(x, J, Faf, af)
#y = idualtree(w, J, Fsf, sf)
#err = x-y
#print(max(abs(err)))
#print('ok')

# Example 3 (Stack Overflow)
N = 128     # Number of samples
t_n = 2     # Signal duration [s]
# Create signal
xc = np.linspace(0, t_n, num=N)
xd = np.linspace(0, t_n, num=32)
sig = np.sin(2*np.pi * 64 * xc[:32]) * (1 - xd)
composite_signal3 = np.concatenate([np.zeros(32), sig[:32], np.zeros(N-32-32)])
lvls = int(np.ceil(np.log2(N)))
Faf, Fsf = FSfarras()       # 1st stage analysis and synthesis filters
af, sf = dualfilt1()        # Remaining stages anal. and synth. filters
x, w = dualtree1D(composite_signal3, lvls, Faf, af)