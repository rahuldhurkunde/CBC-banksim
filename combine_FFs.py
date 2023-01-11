import h5py
import numpy as np
import sys

ninjs = int(sys.argv[1])
nsplits = int(sys.argv[2])

def write_to_hdf(filename, signal_ind, FF, rec_tau, sigmasqs, mass1, mass2, tau0, tau3, best_template):
	with h5py.File(filename, 'w') as hf:
		hf.create_dataset('signal_ind', data=signal_ind)
		hf.create_dataset('FF', data=FF)
		hf.create_dataset('rec_tau', data=rec_tau)
		hf.create_dataset('sigmasqs', data=sigmasqs)
		hf.create_dataset('mass1', data=mass1)
		hf.create_dataset('mass2', data=mass2)
		hf.create_dataset('tau0', data=tau0)
		hf.create_dataset('tau3', data=tau3)
		hf.create_dataset('best_template', data=best_template)
	hf.close()
	print("Succesfully saved ", savefile)
	

signal_ind = []
FF = []
sigmasqs = []
rec_tau = []
mass1 = []
mass2 = []
tau0 = []
tau3 = []
best_template = []
for inj in range(ninjs):
	temp_FF = []
	for split in range(nsplits):
		filename = 'results/H1-FF_%s_%s-0-10.hdf' %(inj, split)
		hf = h5py.File(filename, 'r')
		temp_FF.append(hf['FF'][0])

	max_ind = np.argmax(temp_FF)
	max_file = 'results/H1-FF_%s_%s-0-10.hdf' %(inj, max_ind)
	hm = h5py.File(max_file, 'r')

	signal_ind.append(inj)
	FF.append(hm['FF'][0])
	rec_tau.append(hm['recovered_tau0'][0])
	sigmasqs.append(hm['sigmasqs'][0])
	mass1.append(hm['sg_mass1'][0])
	mass2.append(hm['sg_mass2'][0])
	tau0.append(hm['sg_tau0'][0])
	tau3.append(hm['sg_tau3'][0])
	best_template.append(hm['best_template'][0])
	if (inj % 100 == 0):
		print(inj)

savefile = '%s_FFs.hdf' %ninjs
write_to_hdf(savefile, signal_ind, FF, rec_tau, sigmasqs, mass1, mass2, tau0, tau3, best_template) 	
