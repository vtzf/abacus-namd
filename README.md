# Hefei-NAMD

[Hefei-NAMD](https://github.com/QijingZheng/Hefei-NAMD), an ab-initio nonadiabatic molecular dynamics program, needs the output of DFT program to calculate nonadiabatic coupling (NAC) matrix and perform the following fewest-switches surface hopping (FSSH) or decoherence-induced surface hopping (DISH) simulation. For the purpose of convenient interface between ABACUS and Hefei-NAMD, we provide several python scripts named abacus-namd. In the following part, we take the calculation of hole transfer process at anatase TiO$_2$(001)-(4×1) ADM reconstruction surface as an example. With the help of the python script `NAC.py` provided here, some input files of Hefei-NAMD like `NATXT`and`EIGTXT`can be output. A more convenient and efficient python script`SurfHop.py` is also provided for FSSH/DISH/DCSH calculations.

## Running AIMD

First, the ABACUS input files are needed to perform AIMD NVT and NVE calculation. To output the overlap matrix `data-0-S` and wavefunction file `LOWF_GAMMA_S1.dat` in ouput directory, the extra parameter should be added in `INPUT`file. The parameter `gamma_only` should be set to`1` to only get $\Gamma$-point outputs because current version of Hefei-NAMD interface only supports $\Gamma$-point NAC. A set of input files can be obtained in `example` directory.

```
gamma_only       1
out_wfc_lcao     1
out_mat_hs       1
```

After AIMD calculation, the output directory `OUT.YourSystemName` will contain 3 files: `data-0-H`, `data-0-S` and `LOWF_GAMMA_S1.dat`, which represent Hamiltonian, overlap and wavefunction respectively.

## Get Input Parameters

There are some simple python scripts under `src` directory. To run there scripts normally, please prepare the Python 3.9 interpreter. Install the following Python packages required:

* NumPy
* SciPy
* Numba (only needed for `SurfHop.py`)
* MPI4py >= 3.1.3

Before perform preprocessing and NAMD simulations, some parameters need to be specify in `Args.py`. We list all the parameters needing to be customized and we try to expain some of their meanings afterwards.

```python
# NAMD parameter
# manual input start
dftdir   = '/public/share/zhaojin/tuyy/abacus/sh/OUT.autotest1/'
namddir  = '../namd_test/' # output NAMD output file in namddir
dt       = 1      # MD time step (fs)
start_t  = 1      # start MD step
end_t    = 2000   # end MD step
istart_t = 901    # isample start MD step
iend_t   = 1000   # isample end MD step

LCHARGE  = True   # output atom projected charge density
atom     = [13,26]# atom number of all species (only needed in atomic basis)
orbital  = [27,13]# atomic orbital basis number (only needed in atomic basis)
whichA   = [0,13,14,15] # atom index for projected charge density (starts from 0)

LRANGE   = True   # select range of band, change iband
                  # if not given, LRECOMB specifies energy range
                  # if LRECOMB not given, energy range: [0,nbands]
LHOLE    = True   # Hole/electron transfer
dE       = 2.0    # initial energy from VBM/CBM (eV)

LPHASE   = True   # phase correction

TEMP     = 300    # temperature in Kelvin
NACTIME  = 1000   # time for used NAC (i_end_t-state_t+NACTIME<nstep)
NAMDTIME = 1000   # time for NAMD run
NELM     = 1000   # electron time step (per fs)
NTRAJ    = 5000   # SH trajectories number

LINTERP  = 2      # hamiltonian interpolation algorithm 1,2,3
LTA      = False  # Liouville-Trotter algorithm for small NELM
LSH      = 'FSSH' # run DISH, FSSH or DCSH

LRECOMB  = False  # consider electron-hole recombination

OutMode  = 'text' # output file format 'numpy(.npy)' or 'text'
# manual input end
```

For comparation, the Hefei-NAMD `inp` file is also listed here.

```fortran
&NAMDPARA
  BMIN     = 121    ! bottom band index, starting from 1
  BMAX     = 156    ! top band index
  NBANDS   = 300    ! number of bands

  NSW      = 2000   ! number of ionic steps
  POTIM    = 1.0      ! MD time step
  TEMP     = 300    ! temperature in Kelvin

  NSAMPLE  = 100    ! number of samples
  NAMDTIME = 1000   ! time for NAMD run
  NELM     = 1000   ! electron time step
  NTRAJ    = 5000   ! SH trajectories
  LHOLE    = .TRUE. ! hole/electron SH

  RUNDIR   = "../run"
!  LCPEXT   = .TRUE.
/                   ! **DO NOT FORGET THE SLASH HERE!**
```

* **dftdir**: ABACUS output file folder.
* **namddir**: output file folder of NAMD results.
* **dt**: same as `POTIM` in `inp`. MD time step in fs.
* **start_t & end_t**: start & end MD step for NAC calculation. In `inp` file NSW = end_t - start_t + 1.
* **istart_t & iend_t**: start & end MD step for choosing the init structure in each sample. In `inp` file NSAMPLE = iend_t - istart_t + 1. The `INICON` file can be generated with these two parameters.
* **LCHARGE**: choose whether to obtain charge density of selected atoms or not.
* **LRANGE**: select energy range of band, the reference band is VBM for **LHOLE=True** and CBM for **LHOLE=False**. When considering electron-hole recombination, CBM=VBM+1 and VBM=CBM-1 respectly.
* **LHOLE**: same as `LHOLE` in `inp`.
* **LPHASE**: whether to perform phase correction on NAC or not.
* **NACTIME**: time step for NAC used in NAMD simulations (i_end_t-state_t+NACTIME<nstep)
* **NAMDTIME**: time for NAMD run. If NAMDTIME>NACTIME, the NACs are looped for simulations.
* **LINTERP**: hamiltonian interpolation algorithm in time-dependent wave function evolution. For 1, same as interpolation algorithm in original FSSH program. For 2, same as interpolation algorithm in original DISH program. For 3, interpolation algorithm from [J. Chem. Theory Comput. 2014, 10, 2, 789-804](https://pubs.acs.org/doi/10.1021/ct400934c).
* **LTA**: whether to use Liouville-Trotter algorithm for small NELM simulations or not. Liouville-Trotter algorithm from [J. Chem. Theory Comput. 2014, 10, 2, 789-804](https://pubs.acs.org/doi/10.1021/ct400934c).
* **LSH**: choose DISH, FSSH or DCSH to run NAMD simulations
* **LRECOMB**: whether to consider electron-hole recombination or not
* **OutMode**: output file format 'numpy'(*.npy) or 'text'. For 'text', the output files are same as original Hefei-NAMD program.

## Get NATXT and EIGTXT

A simple python script `NAC.py` under `src` directory is used to conveniently preprocess the ABACUS AIMD outputs and get `NATXT`and`EIGTXT`text files containing nonadiabatic coupling and time-dependent energy levels of selected bands. With these two files, the following NAMD simulation can be performed using original Hefei-NAMD program or `SurfHop.py`.
We list the output files corresponding to input parameters.

`INICON||NATXT||EIGTXT`: **LRANGE=True**
`COUPCAR`: **LRANGE=False**
`DEPHTIME`: **LSH='DISH'/'DCSH'**
`all_en.npy`: ALL

## Phase Correction of Wavefunction

Due to the random phase introduced by different matrix diagonalization methods, the NAC can be quite different under wavefunction with and without phase correction. To overcome this problem, a phase correction operation is available in the perprocessing scripts `NAC.py`, which may cost some time to correct the wavefunction. Here we show the NAC matrix of one anatase TiO$_2$(001)-(4×1) ADM structure and some selected NAC matrix elements in 500 fs evolution. After phase correction of wavefunction, no random jumps during the time evolution of NAC happen.

![NAC matrix with and without phase correction](https://imgur.com/1GJncl7.png)

We list the output files corresponding to input parameters:

`phase_m.npy`: **LPHASE=True**

## Obtain Charge Density of Selected Atoms

In some calculations of carrier migration of hefei-NAMD, the time-dependent distributions of charge carrier is needed, which need charge density projected on each atom to calculate. For example, in VASP output file, `PROCAR` is generated to store atom-projected charge density. In ABACUS instead, the perprocessing scripts `NAC.py` provide option to calculate atom-projected charge density from wavefunction and store it in `all_wht.npy`. With this option, the spatial distribution of carrier migration can be obtained from the postprocessing of output file. Here we show the time-dependent atom-projected charge (hole) density on ridge together with evolution of hole energy and distribution rate on ridge in the 1 ps hole transfer process. Phase correction of wavefunction makes the hole transfer results different.

![Hole transfer process with and without phase correction](https://imgur.com/SrYhG8r.png)

We list the output files corresponding to input parameters:

`all_wht.npy`: **LCHARGE=True**

## Perform NAMD Simulations

After calculation of NAC and some other preprocessing by `NAC.py`, We can finish the following NAMD simulation by using original Hefei-NAMD program because it provides native interface to read `NATXT`, `EIGTXT`and`COUPCAR` files to do surface hopping calculations. An another way is to just use the python scripts `SurfHop.py` in `src` directory instead. `SurfHop.py` integrates several hamiltonian interpolation algorithms in time-dependent wave function evolution and supports FSSH/DISH/DCSH electron hopping algorithms same as original Hefei-NAMD program. The `MPI4py` and `Numba` python library are used to accelerate numerical calculations and make full use of computational resources. By setting parameters in `Args.py`, the NAMD simulations can be easily performed. Here we show a 5 ps FSSH/DCSH/DISH simulation of the hole relaxation with holes initially located on VBM-2.0 eV in anatase TiO$_2$(001)-(4×1) ADM reconstruction surface.

![Hole transfer process with and without phase correction](https://imgur.com/RPJ09jF.png)

We list the output files corresponding to input parameters:

`sh_*.npy`: **OutMode='numpy'**
`SHPROP.*||PSICT.*`: **OutMode='text'**

## Consider Electron-Hole Recombination

When electron-hole recombination is considered in a long-time simulation (usually with decoherence effect), `SurfHop.py` provides a option whether to record the recombination moment or not. recombination rate same as data in `RECOMB.*` can then be derived.Here we show a 1 ns FSSH/DCSH/DISH simulation of the electron-hole recombination with holes initially located on VBM-0.1 eV in anatase TiO$_2$(001)-(4×1) ADM reconstruction surface.

![Hole transfer process with and without phase correction](https://imgur.com/sayyKsE.png)

We list the output files corresponding to input parameters:

`pop_rb.npy`: **OutMode='numpy'** and **LRECOMB=True**
`RECOMB.*`: **OutMode='text'** and **LRECOMB=True**

## Performance Reference

`CPU`: 2nd Gen AMD EPYC™ 7702, 128 cores
`Parameter`:
**NSAMPLE**=100
**NAMDTIME**=5000
**NELM**=1000 (without **LTA**)
**NTRAJ**=5000
**nbands**=36
`NAMD Running Time`:
**FSSH**: 15 seconds/sample
**DISH**: 10 minutes/sample
**DCSH**: 10 minutes/sample

`CPU`: 2nd Gen AMD EPYC™ 7702, 128 cores×2
`Parameter`:
**NSAMPLE**=1
**NAMDTIME**=1000000
**NELM**=10 (with **LTA**)
**NTRAJ**=100000
**nbands**=4
`NAMD Running Time`:
**FSSH**: 6 minutes/sample
**DISH**: 80 minutes/sample
**DCSH**: 90 minutes/sample

Notes:

- All the python scripts should move to the same directory to run because they are interdependent.
- The current Hefei-NAMD preprocess script can only use in ABACUS calculation under lcao basis. So the `basis_type` should be set to `lcao`.
