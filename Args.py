import numpy as np
import os

# NAMD parameter
# manual input start
dftdir   = '../OUT.autotest/'
namddir  = '../namd_test/'
dt       = 1      # MD time step fs
start_t  = 1      # start MD_* dir
end_t    = 2000   # end MD_* dir
istart_t = 901    # isample start MD_* dir
iend_t   = 1000   # isample start MD_* dir

LCHARGE  = True   # output atom projected charge density
atom     = [13,26]# atom number of all species
orbital  = [27,13]# atomic orbital basis number
whichA   = [0,13,14,15] # atom index (starts from 0)

LRANGE   = True   # select range of band, change iband
LHOLE    = False   # Hole transfer
dE       = 2.0    # initial energy from VBM/CBM (eV)

LPHASE   = True   # phase correction

TEMP     = 400    # temperature in Kelvin
NACTIME  = 1000   # time for used NAC
NAMDTIME = 1000   # time for NAMD run
NELM     = 10     # electron time step (per fs)
NTRAJ    = 5000   # SH trajectories

LINTERP  = 2      # interpolation algorithm
LTA      = True   # Liouville-Trotter algorithm
LDISH    = True   # run DISH
# manual input end

# constants
Ry2eV = 13.605698065894
hbar = 0.6582119281559802 # reduced Planck constant eV/fs
eV = 1.60217733E-19
Kb_eV = 8.6173857E-5
Kb = Kb_eV*eV
KbT = kbT = Kb_eV * TEMP

# parameters check
if not os.path.exists(namddir):
    os.makedirs(namddir)

# preprocess
nsample = iend_t-istart_t+1
nstep = end_t-start_t+1
nbands = int((open(dftdir+'MD_%d/LOWF_GAMMA_S1.dat'%(start_t)).readline()).split()[0])
band_s = 1
band_e = nbands
iband_s = 1
iband_e = nbands
ibands = iband_e-iband_s+1
edt = dt/NELM

orb_idx = np.zeros((sum(atom)+1),dtype=int)
atom_idx = np.cumsum([0]+atom)
for i in range(len(atom)):
    orb_idx[atom_idx[i]+1:atom_idx[i+1]+1]=orbital[i]
orb_idx = np.cumsum(orb_idx)

whichO = []
for i in whichA:
    whichO1 = range(orb_idx[i],orb_idx[i+1])
    for j in whichO1:
        whichO.append(j)

