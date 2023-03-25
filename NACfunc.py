import numpy as np
from scipy.integrate import cumtrapz
from scipy.signal import correlate
from scipy.optimize import curve_fit
import Args

def LoadDataNull(step,nsend,Type):
    return np.zeros([step]+nsend,dtype=Type)


# Energy read/write
def ReadE(idx,data):
    energy = data[idx,0]*Args.Ry2eV
    occ = data[idx,1]

    band_vbm = 0
    idx_s = 0
    idx_e = Args.nbands
    while True:
        if ((idx_e-idx_s)<=1):
            band_vbm = idx_e
            break
        band_vbm = int((idx_s+idx_e)/2)
        if (occ[band_vbm]<1e-10):
            idx_e = band_vbm
        else:
            idx_s = band_vbm
    if Args.LHOLE:
        if Args.LRECOMB:
            return np.concatenate((energy,[1],[band_vbm+1]))
        else:
            return np.concatenate((energy,[1],[band_vbm]))
    else:
        if Args.LRECOMB:
            return np.concatenate((energy,[nband],[band_vbm]))
        else:
            return np.concatenate((energy,[nband],[band_vbm+1]))


def gaussian(x,c):
    return np.exp(-x**2/(2*c**2))


def Dephase(energy):
    T = np.arange(Args.nstep-1)*Args.dt
    matrix = np.zeros((energy.shape[1],energy.shape[1]),dtype=float)
    for ii in range(energy.shape[1]):
        for jj in range(ii):
            Et = energy[:,ii]-energy[:,jj]
            Et -= np.average(Et)
            Ct = correlate(Et,Et)[Args.nstep:]/Args.nstep
            Gt = cumtrapz(Ct,dx=Args.dt,initial=0)
            Gt = cumtrapz(Gt,dx=Args.dt,initial=0)
            Dt = np.exp(-Gt/Args.hbar**2)
            popt,pcov = curve_fit(gaussian,T,Dt)
            matrix[ii,jj] = popt[0]
            matrix[jj,ii] = matrix[ii,jj]

    return matrix


# Energy read/write, obtain iband parameters
# save all_en.npy, EIGTXT, INICON
def ReadE1(idx,data):
    energy = data[idx,0]*Args.Ry2eV
    occ = data[idx,1]

    band_vbm = 0
    idx_s = 0
    idx_e = Args.nbands
    while True:
        if ((idx_e-idx_s)<=1):
            band_vbm = idx_e
            break
        band_vbm = int((idx_s+idx_e)/2)
        if (occ[band_vbm]<1e-10):
            idx_e = band_vbm
        else:
            idx_s = band_vbm

    if Args.LHOLE:
        band_init = 0
        idx_s = 0
        idx_e = band_vbm
        while True:
            if ((idx_e-idx_s)<=1):
                band_init = idx_s
                break
            band_init = int((idx_s+idx_e)/2)
            if (energy[band_vbm-1]-energy[band_init]>Args.dE):
                idx_s = band_init
            else:
                idx_e = band_init
        if Args.LRECOMB:
            return np.concatenate((energy,[band_init],[band_vbm+1]))
        else:
            return np.concatenate((energy,[band_init],[band_vbm]))
    else:
        band_init = nband
        idx_s = band_vbm
        idx_e = nband
        while True:
            if ((idx_e-idx_s)<=1):
                band_init = idx_e
                break
            band_init = int((idx_s+idx_e)/2)
            if (energy[band_init]-energy[band_vbm]>Args.dE):
                idx_e = band_init
            else:
                idx_s = band_init
        if Args.LRECOMB:
            return np.concatenate((energy,[band_init],[band_vbm]))
        else:
            return np.concatenate((energy,[band_init],[band_vbm+1]))


def SaveE(savename,energy):
    np.save(Args.namddir+savename,energy[:,:-2])
    iband_s = 1
    iband_e = Args.nbands
    if Args.LHOLE:
        iband_s = int(np.min(energy[:,-2]))
        iband_e = int(np.max(energy[:,-1]))
    else:
        iband_e = int(np.max(energy[:,-2]))
        iband_s = int(np.min(energy[:,-1]))

    with open(Args.namddir+'INICON','w') as f:
        for i in range(Args.istart_t-Args.start_t,Args.iend_t+1-Args.start_t):
            f.write('%3d%5d\n'%(i+1,int(energy[i,-2])))
    with open(Args.namddir+'bandrange.dat','w') as f:
        f.write('%d %d\n'%(iband_s,iband_e))
    np.savetxt(Args.namddir+'EIGTXT',energy[:,iband_s-1:iband_e])
    if Args.LSH != 'FSSH':
        matrix = Dephase(energy[:,iband_s-1:iband_e])
        np.savetxt(Args.namddir+"DEPHTIME",matrix)


# elgenvector/value, overlap read/write
def LoadDataOrbital(step,Type,myid,nidx,idx_range):
    with open(Args.dftdir+'/LOWF_GAMMA_S1.dat') as f:
        nband = int((f.readline()).split()[0])
        norbital = int((f.readline()).split()[0])
        nline = norbital//5 if norbital%5==0 else norbital//5+1
        orbital = np.zeros((nidx[myid],nband,norbital),dtype=Type)
        e_occ = np.zeros((nidx[myid],2,nband),dtype=Type)
        f.seek(0)
        for i in range((Args.start_t+idx_range[myid])*((nline+3)*nband+2)):
            next(f)
        for i in range(nidx[myid]):
            for ii in range(2):
                next(f)
            for j in range(nband):
                next(f)
                e_occ[i,0,j] = float((f.readline()).split()[0])
                e_occ[i,1,j] = float((f.readline()).split()[0])
                for k in range(nline-1):
                    orbital[i,j,k*5:(k+1)*5] \
                    = np.array(f.readline().split(),dtype=Type)
                orbital[i,j,(nline-1)*5:] \
                = np.array(f.readline().split(),dtype=Type)
    
    vec = np.ascontiguousarray(np.swapaxes(orbital,1,2))

    return e_occ, vec


def LoadDataOlp(step,Type,myid,nidx,idx_range):
    N = Args.norbital
    olp = np.zeros((nidx[myid],N,N),dtype=Type)
    olp_t = np.zeros((N,N),dtype=Type)
    idx = np.zeros((N,),dtype=int)
    idx[0] = 1
    with open(Args.dftdir+'/data-0-S') as f:
        for i in range((Args.start_t+idx_range[myid])*N):
            next(f)
        for i in range(nidx[myid]):
            for j in range(N):
                olp_t[j,j:] = np.array(\
                              (f.readline().split())[idx[j]:],dtype=Type)
                olp_t[j,j] /= 2

            olp[i] = olp_t + olp_t.copy().T

    return olp


# charge density read/write, save all_wht.npy
def CDInfo(myid,idx,data,vec,olp):
    All = vec[idx]*np.dot(olp[idx],vec[idx])
    CAll = np.sum(All,axis=0)
    CPart = np.sum(All[Args.whichO],axis=0)

    return CPart/CAll


def SaveCD(savename,data):
    np.save(Args.namddir+savename,data)


# phase correction, save phase_m.npy
def Phase(vec1,vec2,olp1,olp2):
    olp12 = (olp1+olp2)/2
    phase = np.zeros((Args.ibands))
    vec_olp = np.dot(vec1.T,olp12)
    for i in range(Args.ibands):
        phase[i] = np.dot(vec_olp[i],vec2[:,i])

    return np.sign(phase).astype('int32')


def PhaseInfo(myid,idx,data,vec,olp):
    if myid == 0 and idx == 0:
        vec1 = vec[idx+1]
        olp1 = olp[idx+1]
        phase = Phase(vec1,vec1,olp1,olp1)
    else:
        vec1 = vec[idx]
        vec2 = vec[idx+1]
        olp1 = olp[idx]
        olp2 = olp[idx+1]
        phase = Phase(vec1,vec2,olp1,olp2)

    return phase


def PhaseMatrix(phase):
    phase_m = np.zeros((Args.nstep-1,Args.ibands,Args.ibands),dtype='int32')
    
    p0 = phase[0]
    for i in range(1,Args.nstep):
        p1 = p0*phase[i]
        phase_m[i-1] = np.dot(p0.reshape(-1,1),p1.reshape(1,-1))
        p0[:] = p1

    return phase_m


def SavePhase(savename,phase):
    phase_m = PhaseMatrix(phase)
    np.save(Args.namddir+savename,phase_m)


# NAC with/without phase correction, save NATXT
def LoadDataPhase(step,nsend,Type):
    return np.load(Args.namddir+'/phase_m.npy')


def NAC(vec1,vec2,olp1,olp2):
    olp12 = (olp1+olp2)/2
    nac = np.dot(np.dot(vec1.T,olp12),vec2)\
          -np.dot(np.dot(vec2.T,olp12),vec1)

    return nac


def NACInfo(myid,idx,data,vec,olp):
    vec1 = vec[idx]
    vec2 = vec[idx+1]
    olp1 = olp[idx]
    olp2 = olp[idx+1]
    nac = NAC(vec1,vec2,olp1,olp2)
    return nac


def NACPhase(vec1,vec2,olp1,olp2,phase):
    olp12 = (olp1+olp2)/2
    nac = (np.dot(np.dot(vec1.T,olp12),vec2))*phase\
          -(np.dot(np.dot(vec2.T,olp12),vec1))*phase.T

    return nac


def NACPhaseInfo(myid,idx,phase,vec,olp):
    vec1 = vec[idx]
    vec2 = vec[idx+1]
    olp1 = olp[idx]
    olp2 = olp[idx+1]
    phase0 = phase[idx]
    nac = NACPhase(vec1,vec2,olp1,olp2,phase0)
    return nac


def SaveNAC(savename,data):
    np.savetxt(Args.namddir+savename,data.reshape(Args.nstep-1,-1))


def SaveCOUP(savename,data):
    energy = np.load(Args.namddir+'/all_en.npy')
    buf = np.zeros((Args.nstep,Args.nbands+1,Args.nbands))
    buf[0,0,0] = Args.nbands*(Args.nbands+1)*8
    buf[0,0,1] = Args.nbands
    buf[0,0,2] = Args.nstep
    buf[0,0,3] = Args.dt
    buf[1:,Args.nbands,:] = energy[0:-1]
    buf[1:,0:Args.nbands,:] = data
    buf.tofile(Args.namddir+savename)
