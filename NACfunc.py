import numpy as np
from scipy.integrate import cumtrapz
from scipy.signal import correlate
from scipy.optimize import curve_fit
import Args

def LoadDataNull(step,nsend,Type):
    return np.zeros([step]+nsend,dtype=Type)


# Energy read/write
def ReadE(idx,data):
    line = open(Args.dftdir+'/MD_%d/LOWF_GAMMA_S1.dat'%(idx)).readlines()

    nband = int(line[0].split()[0])
    norbital = int(line[1].split()[0])
    nline = norbital//5+1+3

    energy = np.array([line[i*nline+3].split()[0] \
             for i in range(nband)], dtype=float)*Args.Ry2eV

    return energy


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


def SaveE(savename,energy):
    np.save(Args.namddir+savename,energy)
    np.savetxt(Args.namddir+'EIGTXT',energy)
    if Args.LDISH:
        matrix = Dephase(energy)
        np.savetxt(Args.namddir+"DEPHTIME",matrix)


# Energy read/write, obtain iband parameters
# save all_en.npy, EIGTXT, INICON
def ReadE1(idx,data):
    line = open(Args.dftdir+'/MD_%d/LOWF_GAMMA_S1.dat'%(idx)).readlines()

    nband = int(line[0].split()[0])
    norbital = int(line[1].split()[0])
    nline = norbital//5+1+3

    energy = np.array([line[i*nline+3].split()[0] \
             for i in range(nband)], dtype=float)*Args.Ry2eV
    occ = np.array([line[i*nline+4].split()[0] \
          for i in range(nband)], dtype=float)

    #band_vbm = 0
    #for i in range(nband):
    #    if occ[i]<1e-10:
    #        band_vbm = i
    #        break

    #if Args.LHOLE:
    #    band_init = 0
    #    for i in range(band_vbm):
    #        if energy[band_vbm-1]-energy[i]>Args.dE \
    #        and energy[band_vbm-1]-energy[i+1]<Args.dE:
    #            band_init = i
    #            break
    #    return np.concatenate((energy,[band_init],[band_vbm]))
    #else:
    #    band_init = band_vbm
    #    for i in range(band_vbm,nband):
    #        if energy[i-1]-energy[band_vbm]<Args.dE \
    #        and energy[i]-energy[band_vbm]>Args.dE:
    #            band_init = i
    #            break
    #    return np.concatenate((energy,[band_init],[band_vbm+1]))

    band_vbm = 0
    idx_s = 0
    idx_e = nband
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
        return np.concatenate((energy,[band_init],[band_vbm+1]))


def SaveE1(savename,energy):
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
    if Args.LDISH:
        matrix = Dephase(energy[:,iband_s-1:iband_e])
        np.savetxt(Args.namddir+"DEPHTIME",matrix)


# elgenvector/overlap read/write
def ReadOrbital(Dir,Band_s,Band_e):
    line = open(Dir+'/LOWF_GAMMA_S1.dat').readlines()

    nband = int(line[0].split()[0])
    norbital = int(line[1].split()[0])
    nline = norbital//5+1+3

    orbital_tmp = [line[j].split() for i in range(nband) \
                  for j in range(i*nline+5,(i+1)*nline+2)]
    orbital = np.array([i for j in orbital_tmp \
              for i in j], dtype=float).reshape(nband,norbital)

    return orbital[Band_s-1:Band_e].T


def ReadMatrix(Dir,Name):
    line = open(Dir+'/'+Name).readlines()
    N = int(line[0].split()[0])
    idx = np.zeros((N,),dtype=int)
    idx[0] = 1
    data = [np.array(line[i].split()[idx[i]:],dtype=float) \
            for i in range(N)]

    mat_full = np.zeros((N,N))

    for i in range(N):
        mat_full[i,i:] = data[i]
    mat_full = mat_full + mat_full.copy().T
    for i in range(N):
        mat_full[i,i] /= 2

    return mat_full


# charge density read/write, save all_wht.npy
def ChargeDensity(vec,olp):
    All = vec*np.dot(olp,vec)
    CAll = np.sum(All,axis=0)
    CPart = np.sum(All[Args.whichO],axis=0)

    return CPart/CAll


def CDInfo(idx,data):
    vec = ReadOrbital(Args.dftdir+'/MD_'+str(idx),Args.band_s,Args.band_e)
    olp = ReadMatrix(Args.dftdir+'/MD_'+str(idx),'data-0-S')

    CD = ChargeDensity(vec,olp)

    return CD


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


#def Phase(vec1,vec2,olp1,olp2):
#    olp12 = (olp1+olp2)/2
#    phase = np.dot(np.dot(vec1.T,olp12),vec2)
#    phase = np.diagonal(phase)
#
#    return np.sign(phase).astype('int32')


def PhaseInfo(idx,data):
    if idx == Args.start_t:
        vec1 = ReadOrbital(Args.dftdir+'/MD_'+str(idx),Args.iband_s,Args.iband_e)
        olp1 = ReadMatrix(Args.dftdir+'/MD_'+str(idx),'data-0-S')
        phase = Phase(vec1,vec1,olp1,olp1)
    else:
        vec1 = ReadOrbital(Args.dftdir+'/MD_'+str(idx-1),Args.iband_s,Args.iband_e)
        vec2 = ReadOrbital(Args.dftdir+'/MD_'+str(idx),Args.iband_s,Args.iband_e)
        olp1 = ReadMatrix(Args.dftdir+'/MD_'+str(idx-1),'data-0-S')
        olp2 = ReadMatrix(Args.dftdir+'/MD_'+str(idx),'data-0-S')
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


def NACInfo(idx,phase):
    vec1 = ReadOrbital(Args.dftdir+'/MD_'+str(idx),Args.iband_s,Args.iband_e)
    vec2 = ReadOrbital(Args.dftdir+'/MD_'+str(idx+1),Args.iband_s,Args.iband_e)
    olp1 = ReadMatrix(Args.dftdir+'/MD_'+str(idx),'data-0-S')
    olp2 = ReadMatrix(Args.dftdir+'/MD_'+str(idx+1),'data-0-S')
    nac = NAC(vec1,vec2,olp1,olp2)
    return nac


def NACPhase(vec1,vec2,olp1,olp2,phase):
    olp12 = (olp1+olp2)/2
    nac = (np.dot(np.dot(vec1.T,olp12),vec2))*phase\
          -(np.dot(np.dot(vec2.T,olp12),vec1))*phase.T

    return nac


def NACPhaseInfo(idx,phase):
    vec1 = ReadOrbital(Args.dftdir+'/MD_'+str(idx),Args.iband_s,Args.iband_e)
    vec2 = ReadOrbital(Args.dftdir+'/MD_'+str(idx+1),Args.iband_s,Args.iband_e)
    olp1 = ReadMatrix(Args.dftdir+'/MD_'+str(idx),'data-0-S')
    olp2 = ReadMatrix(Args.dftdir+'/MD_'+str(idx+1),'data-0-S')
    nac = NACPhase(vec1,vec2,olp1,olp2,phase)
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
