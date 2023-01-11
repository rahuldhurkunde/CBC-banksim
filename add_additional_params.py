import numpy as np
import h5py
from pycbc import conversions
import sys

filename = sys.argv[1]
if filename == []:
    sys.exit('Please provide a injection (.hdf) file')

else:
    hf = h5py.File(filename, 'r+')
    mass1 = hf['mass1'][:]
    mass2 = hf['mass2'][:]
    f_lower = hf.attrs['f_lower']

    # Assign the injections above values randomly
    ninjs = len(hf['mass1'][:])
    spin1x = np.zeros(ninjs)
    spin1y = np.zeros(ninjs)
    spin2x = np.zeros(ninjs)
    spin2y = np.zeros(ninjs)
	

    #Get distance from chirp_distance
    tau0 = conversions.tau0_from_mass1_mass2(mass1, mass2, f_lower)
    tau3 = conversions.tau3_from_mass1_mass2(mass1, mass2, f_lower)

    # Add remaining params to the injection file
    with hf:
        hf.create_dataset("spin1x", data=spin1x)
        hf.create_dataset("spin1y", data=spin1y)
        hf.create_dataset("spin2x", data=spin2x)
        hf.create_dataset("spin2y", data=spin2y)
        hf.create_dataset("tau0", data=tau0)
        hf.create_dataset("tau3", data=tau3)
    hf.close()

    print('spin1x-1y, spin2x-2y and tau0-tau3 added to', filename)

