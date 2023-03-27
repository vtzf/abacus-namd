#!/usr/bin/env python
#-*- encoding:utf-8 -*-
import Args
import numpy as np
from mpi4py import MPI
from numba import jit

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
nprocs = comm.Get_size()

# integration function
@jit(nopython=True,fastmath=True,cache=False)
def GetHamil(NAC,Energy,ibands,istep,estep):
    wht = estep/Args.NELM
    en = Energy[(istep+1)%Args.NACTIME]*wht + Energy[istep]*(1.0-wht)
    nac = NAC[(istep+1)%Args.NACTIME]*wht + NAC[istep]*(1.0-wht)

    return en, nac


@jit(nopython=True,fastmath=True,cache=False)
def GetHamilDt(NAC,Energy,ibands,istep,estep):
    wht = estep/Args.NELM
    en = Energy[(istep+1)%Args.NACTIME]*wht + Energy[istep]*(1.0-wht)
    if wht<0.5:
        nac = NAC[istep]*(0.5+wht)+NAC[istep-1]*(0.5-wht)
    else:
        nac = NAC[(istep+1)%Args.NACTIME]*(wht-0.5)+NAC[istep]*(1.5-wht)

    return en, nac


@jit(nopython=True,fastmath=True,cache=False)
def GetHamil2(NAC,Energy,ibands,istep,estep):
    wht = estep/Args.NELM/2
    en = Energy[istep]+(Energy[(istep+1)%(Args.NACTIME)]-Energy[istep-1])*wht
    nac = NAC[istep]+(NAC[(istep+1)%Args.NACTIME]-NAC[istep-1])*wht

    return en, nac


# Liouville-Trotter algorithm
@jit(nopython=True,fastmath=True,cache=False)
def LTA(en,nac,ibands,psi1):
    cos_nac = np.cos(-nac*(Args.edt/(4*Args.dt)))
    sin_nac = np.sin(-nac*(Args.edt/(4*Args.dt)))
    exp_en = np.exp(en*(Args.edt/(1j*Args.hbar)))
    for i in range(ibands):
        for j in range(i+1,ibands):
            psi_i = psi1[i]
            psi_j = psi1[j]
            psi1[i] =  cos_nac[i,j]*psi_i+sin_nac[i,j]*psi_j
            psi1[j] = -sin_nac[i,j]*psi_i+cos_nac[i,j]*psi_j
    psi1 *= exp_en
    for i in range(ibands-1,-1,-1):
        for j in range(ibands-1,i,-1):
            psi_i = psi1[i]
            psi_j = psi1[j]
            psi1[i] =  cos_nac[i,j]*psi_i+sin_nac[i,j]*psi_j
            psi1[j] = -sin_nac[i,j]*psi_i+cos_nac[i,j]*psi_j


@jit(nopython=True,fastmath=True,cache=False)
def TimeProp(NAC,Energy,psi0,psi1,psi2,ibands,istep,estep):
    if Args.LINTERP == 1:
        en, nac = GetHamil(NAC,Energy,ibands,istep%(Args.NACTIME),estep)
        if istep==0 and estep==0:
            psi_r = np.ascontiguousarray(np.real(psi0))
            psi_i = np.ascontiguousarray(np.imag(psi0))
            nac_psi = np.dot(nac,psi_r)+1j*np.dot(nac,psi_i)
            psi1[:] = psi0 + ((en*psi0)/(1j*Args.hbar)-nac_psi/(2*Args.dt))*Args.edt
            psi2[:] = psi1
        else:
            psi_r = np.ascontiguousarray(np.real(psi1))
            psi_i = np.ascontiguousarray(np.imag(psi1))
            nac_psi = np.dot(nac,psi_r)+1j*np.dot(nac,psi_i)
            psi2[:] = psi0 + ((en*psi1)/(1j*Args.hbar)-nac_psi/(2*Args.dt))*(2*Args.edt)
            psi0[:] = psi1
            psi1[:] = psi2

    elif Args.LINTERP == 2:
        en, nac = GetHamilDt(NAC,Energy,ibands,istep%(Args.NACTIME),estep)
        if istep==0 and estep==0:
            psi_r = np.ascontiguousarray(np.real(psi0))
            psi_i = np.ascontiguousarray(np.imag(psi0))
            nac_psi = np.dot(nac,psi_r)+1j*np.dot(nac,psi_i)
            psi1[:] = psi0 + ((en*psi0)/(1j*Args.hbar)-nac_psi/(2*Args.dt))*Args.edt
            psi2[:] = psi1
        else:
            psi_r = np.ascontiguousarray(np.real(psi1))
            psi_i = np.ascontiguousarray(np.imag(psi1))
            nac_psi = np.dot(nac,psi_r)+1j*np.dot(nac,psi_i)
            psi2[:] = psi0 + ((en*psi1)/(1j*Args.hbar)-nac_psi/(2*Args.dt))*(2*Args.edt)
            psi0[:] = psi1
            psi1[:] = psi2

    elif Args.LINTERP == 3:
        if istep==0:
            en, nac = GetHamil(NAC,Energy,ibands,istep%(Args.NACTIME),estep)
            if estep==0:
                psi_r = np.ascontiguousarray(np.real(psi0))
                psi_i = np.ascontiguousarray(np.imag(psi0))
                nac_psi = np.dot(nac,psi_r)+1j*np.dot(nac,psi_i)
                psi1[:] = psi0 + ((en*psi0)/(1j*Args.hbar)-nac_psi/(2*Args.dt))*Args.edt
                psi2[:] = psi1
            else:
                psi_r = np.ascontiguousarray(np.real(psi1))
                psi_i = np.ascontiguousarray(np.imag(psi1))
                nac_psi = np.dot(nac,psi_r)+1j*np.dot(nac,psi_i)
                psi2[:] = psi0 + ((en*psi1)/(1j*Args.hbar)-nac_psi/(2*Args.dt))*(2*Args.edt)
                psi0[:] = psi1
                psi1[:] = psi2
        else:
            en, nac = GetHamil2(NAC,Energy,ibands,istep%(Args.NACTIME),estep)
            psi_r = np.ascontiguousarray(np.real(psi1))
            psi_i = np.ascontiguousarray(np.imag(psi1))
            nac_psi = np.dot(nac,psi_r)+1j*np.dot(nac,psi_i)
            psi2[:] = psi0 + ((en*psi1)/(1j*Args.hbar)-nac_psi/(2*Args.dt))*(2*Args.edt)
            psi0[:] = psi1
            psi1[:] = psi2


@jit(nopython=True,fastmath=True,cache=False)
def TimePropLTA(NAC,Energy,psi1,ibands,istep,estep):
    if Args.LINTERP == 1:
        en, nac = GetHamil(NAC,Energy,ibands,istep%(Args.NACTIME),estep)

    elif Args.LINTERP == 2:
        en, nac = GetHamilDt(NAC,Energy,ibands,istep%(Args.NACTIME),estep)

    elif Args.LINTERP == 3:
        if istep==0:
            en, nac = GetHamil(NAC,Energy,ibands,istep%(Args.NACTIME),estep)
        else:
            en, nac = GetHamil2(NAC,Energy,ibands,istep%(Args.NACTIME),estep)

    LTA(en,nac,ibands,psi1)


# DISH function
@jit(nopython=True,fastmath=True,cache=False)
def dishhop(
    ibands,DephaseR,Energy,psi1,decmoment,
    index,state,isrecomb,norecomb
):
    pop_t = np.ascontiguousarray(np.real(np.conj(psi1)*psi1))
    decotime = 1/np.dot(DephaseR,pop_t)

    which = -1
    np.random.shuffle(index)
    for i in range(ibands):
        j = index[i]
        if (decotime[j]<=decmoment[j]):
            which = j
            decmoment[which] = 0
            break
    decmoment += Args.dt

    if (which>=0):
        prop = pop_t[which]
        if Args.LHOLE:
            dE = Energy[state]-Energy[which]
            if dE>0:
                prop *= np.exp(-dE/Args.KbT)
        else:
            dE = Energy[which]-Energy[state]
            if dE>0:
                prop *= np.exp(-dE/Args.KbT)    

        prop0 = np.random.rand()
        if (prop0<=prop):
            psi1[:] = 0
            psi1[which] = 1
            if Args.LRECOMB:
                if Args.LHOLE:
                    if (which==ibands-1 and norecomb):
                        isrecomb = True
                        norecomb = False
                else:
                    if (which==0 and norecomb):
                        isrecomb = True
                        norecomb = False
            state = which
        else:
            psi1[which] = 0
            psi1 /= np.linalg.norm(psi1)

    return state, isrecomb, norecomb


@jit(nopython=True,fastmath=True,cache=False)
def dish(
    ntraj,ibands,state_s,nac,
    energy,DephaseR,pop_sh_p,pop_rb_p
):
    psi0 = np.zeros((ibands),dtype=np.complex128)
    psi1 = np.zeros((ibands),dtype=np.complex128)
    psi2 = np.zeros((ibands),dtype=np.complex128)
    decmoment = np.zeros((ibands),dtype=np.float64)
    index = np.arange(ibands)

    for i in range(ntraj):
        isrecomb = False
        norecomb = True
        psi0[:] = 0
        psi1[:] = 0
        psi2[:] = 0
        state = state_s
        psi0[state] = 1
        psi1[state] = 1
        decmoment[:] = 0
        for j in range(Args.NAMDTIME):
            pop_sh_p[j,state] += 1
            for k in range(Args.NELM):
                if Args.LTA:
                    TimePropLTA(nac,energy,psi1,ibands,j,k)
                else:
                    TimeProp(nac,energy,psi0,psi1,psi2,ibands,j,k)
            if Args.LTA:
                state, isrecomb, norecomb \
                = dishhop(ibands,DephaseR,energy[j%(Args.NACTIME)],\
                          psi1,decmoment,index,state,isrecomb,norecomb)
            else:
                state, isrecomb, norecomb \
                = dishhop(ibands,DephaseR,energy[j%(Args.NACTIME)],\
                          psi2,decmoment,index,state,isrecomb,norecomb)
                psi0[:] = psi1
                psi1[:] = psi2
            if Args.LRECOMB:
                if (isrecomb and not norecomb):
                    pop_rb_p[j:Args.NAMDTIME] += 1
                    isrecomb = False

    pop_sh_p /= Args.NTRAJ
    if Args.LRECOMB:
        pop_rb_p /= Args.NTRAJ


@jit(nopython=True,fastmath=True,cache=False)
def dish_e(e_sh,pop_sh,energy):
    for j in range(Args.NAMDTIME):
        pop_sh_t = (pop_sh[j]).astype(np.float64)
        e_sh[j] = np.dot(pop_sh_t,energy[j%Args.NACTIME])


def MPIdish(
    ibands,state_s,ntraj,nac,energy,
    DephaseR,pop_sh,pop_rb,e_sh
):
    pop_sh_p = np.zeros((Args.NAMDTIME,ibands),dtype=np.float32)
    if Args.LRECOMB:
        pop_rb_p = np.zeros((Args.NAMDTIME),dtype=np.float32)
        dish(ntraj,ibands,state_s,nac,energy,DephaseR,pop_sh_p,pop_rb_p)
        comm.Reduce(pop_rb_p,pop_rb,op=MPI.SUM,root=0)
    else:
        pop_rb_p = np.zeros((0),dtype=np.float32)
        dish(ntraj,ibands,state_s,nac,energy,DephaseR,pop_sh_p,pop_rb_p)
    comm.Reduce(pop_sh_p,pop_sh,op=MPI.SUM,root=0)
    if myid == 0:
        dish_e(e_sh,pop_sh,energy)


# FSSH functions
@jit(nopython=True,fastmath=True,cache=False)
def fsshhop(ibands,psi,NAC,Energy,state,isrecomb,norecomb):
    density = np.real(np.conj(psi[state])*psi)
    prop = (density*NAC[state])/density[state]
    prop = np.maximum(prop,0)
    if Args.LHOLE:
        dE = Energy[state]-Energy[0:state]
        prop[0:state] *= np.exp(-dE/Args.KbT)
    else:
        dE = Energy[state:ibands]-Energy[state]
        prop[state:ibands] *= np.exp(-dE/Args.KbT)
    
    prop0 = np.random.rand()

    propall = np.zeros((ibands+1),dtype=np.float64)
    propall[1:] = np.cumsum(prop)

    if (prop0<=propall[ibands]):
        idx_s = 0
        idx_e = ibands
        while True:
            if ((idx_e-idx_s)<=1):
                state = idx_s
                break
            state = int((idx_s+idx_e)/2)
            if (prop0<propall[state]):
                idx_e = state
            else:
                idx_s = state

    if Args.LRECOMB:
        if Args.LHOLE:
            if (state==ibands-1 and norecomb):
                isrecomb = True
                norecomb = False
        else:
            if (state==0 and norecomb):
                isrecomb = True
                norecomb = False

    return state, isrecomb, norecomb


@jit(nopython=True,fastmath=True,cache=False)
def fssh_psi(ibands,state_s,psi_p,nac,energy):
    psi0 = np.zeros((ibands),dtype=np.complex128)
    psi1 = np.zeros((ibands),dtype=np.complex128)
    psi2 = np.zeros((ibands),dtype=np.complex128)
    psi0[state_s] = 1
    psi1[state_s] = 1

    for j in range(Args.NAMDTIME):
        psi_p[j] = psi1
        for k in range(Args.NELM):
            if Args.LTA:
                TimePropLTA(nac,energy,psi1,ibands,j,k)
            else:
                TimeProp(nac,energy,psi0,psi1,psi2,ibands,j,k)


@jit(nopython=True,fastmath=True,cache=False)
def fssh_pop(ntraj,ibands,state_s,psi_p,pop_sh_p,pop_rb_p,nac,energy):
    for i in range(ntraj):
        isrecomb = False
        norecomb = True
        state = state_s
        for j in range(Args.NAMDTIME):
            pop_sh_p[j,state] += 1
            state,isrecomb,norecomb\
            = fsshhop(ibands,psi_p[j],nac[j%(Args.NACTIME)],\
                      energy[j%(Args.NACTIME)],state,isrecomb,norecomb)
            if Args.LRECOMB:
                if (isrecomb and not norecomb):
                    pop_rb_p[j:Args.NAMDTIME] += 1
                    isrecomb = False
                
    pop_sh_p /= Args.NTRAJ
    if Args.LRECOMB:
        pop_rb_p /= Args.NTRAJ


def fssh(ntraj,ibands,state_s,nac,energy,psi_p,pop_sh_p,pop_rb_p):
    if myid == 0:
        fssh_psi(ibands,state_s,psi_p,nac,energy)
    comm.Bcast(psi_p,root=0)
    fssh_pop(ntraj,ibands,state_s,psi_p,pop_sh_p,pop_rb_p,nac,energy)


@jit(nopython=True,fastmath=True,cache=False)
def fssh_e(psi_p,psi,pop_psi,pop_sh,e_psi,e_sh,energy):
    psi[:] = psi_p
    pop_psi[:] = np.real(np.conj(psi_p)*psi_p)
    for j in range(Args.NAMDTIME):
        e = energy[j%Args.NACTIME]
        e_psi[j] = np.dot(pop_psi[j],e)
        pop_sh_t = (pop_sh[j]).astype(np.float64)
        e_sh[j] = np.dot(pop_sh_t,e)


def MPIfssh(
    ibands,state_s,ntraj,nac,energy,
    psi,pop_psi,e_psi,pop_sh,pop_rb,e_sh
):
    psi_p = np.zeros((Args.NAMDTIME,ibands),dtype=np.complex128)
    pop_sh_p = np.zeros((Args.NAMDTIME,ibands),dtype=np.float32)
    if Args.LRECOMB:
        pop_rb_p = np.zeros((Args.NAMDTIME),dtype=np.float32)
        fssh(ntraj,ibands,state_s,nac,energy,psi_p,pop_sh_p,pop_rb_p)
        comm.Reduce(pop_rb_p,pop_rb,op=MPI.SUM,root=0)
    else:
        pop_rb_p = np.zeros((0),dtype=np.float32)
        fssh(ntraj,ibands,state_s,nac,energy,psi_p,pop_sh_p,pop_rb_p)
    comm.Reduce(pop_sh_p,pop_sh,op=MPI.SUM,root=0)
    if myid == 0:
        fssh_e(psi_p,psi,pop_psi,pop_sh,e_psi,e_sh,energy)


# DCSH functions
@jit(nopython=True,fastmath=True,cache=False)
def dcsh_psi(psi1,DephaseR,state):
    pop_t = np.ascontiguousarray(np.real(np.conj(psi1)*psi1))
    decotime = 1/np.dot(DephaseR,pop_t)
    psi1_t = psi1*np.exp(-Args.dt/decotime)
    pop_t1 = 1-np.real(np.dot(np.conj(psi1_t),psi1_t) \
              -np.conj(psi1_t[state])*psi1_t[state])
    psi1_norm = psi1[state]*np.sqrt(pop_t1/pop_t[state])
    psi1[:] = psi1_t
    psi1[state] = psi1_norm


@jit(nopython=True,fastmath=True,cache=False)
def dcsh(
    ntraj,ibands,state_s,nac,
    energy,DephaseR,pop_sh_p,pop_rb_p
):
    psi0 = np.zeros((ibands),dtype=np.complex128)
    psi1 = np.zeros((ibands),dtype=np.complex128)
    psi2 = np.zeros((ibands),dtype=np.complex128)

    for i in range(ntraj):
        isrecomb = False
        norecomb = True
        psi0[:] = 0
        psi1[:] = 0
        psi2[:] = 0
        state = state_s
        psi0[state] = 1
        psi1[state] = 1
        for j in range(Args.NAMDTIME):
            pop_sh_p[j,state] += 1
            for k in range(Args.NELM):
                if Args.LTA:
                    TimePropLTA(nac,energy,psi1,ibands,j,k)
                else:
                    TimeProp(nac,energy,psi0,psi1,psi2,ibands,j,k)
            if Args.LTA:
                #dcsh_psi(psi1,DephaseR,state)
                state,isrecomb,norecomb \
                = fsshhop(ibands,psi1,nac[j%(Args.NACTIME)],\
                          energy[j%(Args.NACTIME)],state,isrecomb,norecomb)
                dcsh_psi(psi1,DephaseR,state)
            else:
                #dcsh_psi(psi2,DephaseR,state)
                state, isrecomb, norecomb \
                = fsshhop(ibands,psi2,nac[j%(Args.NACTIME)],\
                          energy[j%(Args.NACTIME)],state,isrecomb,norecomb)
                dcsh_psi(psi2,DephaseR,state)
                psi0[:] = psi1
                psi1[:] = psi2
            if Args.LRECOMB:
                if (isrecomb and not norecomb):
                    pop_rb_p[j:Args.NAMDTIME] += 1
                    isrecomb = False

    pop_sh_p /= Args.NTRAJ
    if Args.LRECOMB:
        pop_rb_p /= Args.NTRAJ


@jit(nopython=True,fastmath=True,cache=False)
def dcsh_e(e_sh,pop_sh,energy):
    for j in range(Args.NAMDTIME):
        pop_sh_t = (pop_sh[j]).astype(np.float64)
        e_sh[j] = np.dot(pop_sh_t,energy[j%Args.NACTIME])


def MPIdcsh(
    ibands,state_s,ntraj,nac,energy,
    DephaseR,pop_sh,pop_rb,e_sh
):
    pop_sh_p = np.zeros((Args.NAMDTIME,ibands),dtype=np.float32)
    if Args.LRECOMB:
        pop_rb_p = np.zeros((Args.NAMDTIME),dtype=np.float32)
        dcsh(ntraj,ibands,state_s,nac,energy,DephaseR,pop_sh_p,pop_rb_p)
        comm.Reduce(pop_rb_p,pop_rb,op=MPI.SUM,root=0)
    else:
        pop_rb_p = np.zeros((0),dtype=np.float32)
        dcsh(ntraj,ibands,state_s,nac,energy,DephaseR,pop_sh_p,pop_rb_p)
    comm.Reduce(pop_sh_p,pop_sh,op=MPI.SUM,root=0)
    if myid == 0:
        dcsh_e(e_sh,pop_sh,energy)


def SurfHop():
    inicon = np.loadtxt(Args.namddir+'INICON',dtype=int)
    bandrange = np.loadtxt(Args.namddir+'bandrange.dat',dtype=int)
    iband_s = bandrange[0]
    iband_e = bandrange[1]
    ibands = iband_e - iband_s + 1
    ntraj = (Args.NTRAJ*(myid+1))//nprocs\
          - (Args.NTRAJ*myid)//nprocs

    nac = np.loadtxt(Args.namddir+'NATXT').reshape(-1,ibands,ibands)
    energy = np.loadtxt(Args.namddir+'EIGTXT')

    #Args.nsample = 1
    if myid == 0:
        if Args.LRECOMB:
            pop_rb = np.zeros((Args.nsample,Args.NAMDTIME),dtype=np.float32)
        if Args.LSH != 'FSSH':
            pop_sh = np.zeros((Args.nsample,Args.NAMDTIME,ibands),dtype=np.float32)
            e_sh = np.zeros((Args.nsample,Args.NAMDTIME),dtype=np.float64)
        else:
            psi = np.zeros((Args.nsample,Args.NAMDTIME,ibands),dtype=np.complex128)
            pop_psi = np.zeros((Args.nsample,Args.NAMDTIME,ibands),dtype=np.float64)
            e_psi = np.zeros((Args.nsample,Args.NAMDTIME))
            pop_sh = np.zeros((Args.nsample,Args.NAMDTIME,ibands),dtype=np.float32)
            e_sh = np.zeros((Args.nsample,Args.NAMDTIME))

    for i in range(Args.nsample):
        starttime = MPI.Wtime()
        timeinit = inicon[i,0]-Args.start_t
        bandinit = inicon[i,1]
        state_s = bandinit-iband_s
        if Args.LSH != 'FSSH':
            DephaseT = np.loadtxt(Args.namddir+'DEPHTIME')
            DephaseR = np.zeros((ibands,ibands))
            idx = np.where(DephaseT!=0)
            DephaseR[idx] = 1/DephaseT[idx]
            if myid == 0:
                if Args.LSH == 'DISH':
                    MPIdish(
                        ibands,state_s,ntraj,
                        nac[timeinit:timeinit+Args.NACTIME],
                        energy[timeinit:timeinit+Args.NACTIME],
                        DephaseR,pop_sh[i],
                        pop_rb[i] if Args.LRECOMB else None,
                        e_sh[i]
                    )
                elif Args.LSH == 'DCSH':
                    MPIdcsh(
                        ibands,state_s,ntraj,
                        nac[timeinit:timeinit+Args.NACTIME],
                        energy[timeinit:timeinit+Args.NACTIME],
                        DephaseR,pop_sh[i],
                        pop_rb[i] if Args.LRECOMB else None,
                        e_sh[i]
                    )
            else:
                if Args.LSH == 'DISH':
                    MPIdish(
                        ibands,state_s,ntraj,
                        nac[timeinit:timeinit+Args.NACTIME],
                        energy[timeinit:timeinit+Args.NACTIME],
                        DephaseR,None,None,None
                    )
                elif Args.LSH == 'DCSH':
                    MPIdcsh(
                        ibands,state_s,ntraj,
                        nac[timeinit:timeinit+Args.NACTIME],
                        energy[timeinit:timeinit+Args.NACTIME],
                        DephaseR,None,None,None
                    )
        else:
            if myid == 0:
                MPIfssh(
                    ibands,state_s,ntraj,
                    nac[timeinit:timeinit+Args.NACTIME],
                    energy[timeinit:timeinit+Args.NACTIME],
                    psi[i],pop_psi[i],e_psi[i],pop_sh[i],
                    pop_rb[i] if Args.LRECOMB else None,
                    e_sh[i]
                )
            else:
                 MPIfssh(
                    ibands,state_s,ntraj,
                    nac[timeinit:timeinit+Args.NACTIME],
                    energy[timeinit:timeinit+Args.NACTIME],
                    None,None,None,None,None,None
                )
        endtime = MPI.Wtime()
        if myid == 0:
            print("%s time in sample %d: %.6fs"\
                  %(Args.LSH,i,endtime-starttime))
            if Args.OutMode == 'numpy':
                if Args.LSH != 'FSSH':
                    if Args.LSH == 'DISH':
                        np.save(Args.namddir+'dish_pop_sh.npy',pop_sh)
                        np.save(Args.namddir+'dish_e_sh.npy',e_sh)
                        if Args.LRECOMB:
                            np.save(Args.namddir+'dish_pop_rb.npy',pop_rb)
                    elif Args.LSH == 'DCSH':
                        np.save(Args.namddir+'dcsh_pop_sh.npy',pop_sh)
                        np.save(Args.namddir+'dcsh_e_sh.npy',e_sh)
                        if Args.LRECOMB:
                            np.save(Args.namddir+'dcsh_pop_rb.npy',pop_rb)
                else:
                    np.save(Args.namddir+'fssh_psi.npy',psi)
                    np.save(Args.namddir+'fssh_pop_psi.npy',pop_psi)
                    np.save(Args.namddir+'fssh_e_psi.npy',e_psi)
                    np.save(Args.namddir+'fssh_pop_sh.npy',pop_sh)
                    np.save(Args.namddir+'fssh_e_sh.npy',e_sh)
                    if Args.LRECOMB:
                        np.save(Args.namddir+'fssh_pop_rb.npy',pop_rb)
        
            elif Args.OutMode == 'text':
                for i in range(Args.nsample):
                    with open(Args.namddir+'/SHPROP.%d'%(inicon[i,0]),'w') as f:
                        for j in range(Args.NAMDTIME):
                            f.write('%4d %13.8f'%(j+1,e_sh[i,j]))
                            for k in range(ibands):
                                f.write('%10.6f'%(pop_sh[i,j,k]))
                            f.write('\n')
                    if Args.LRECOMB:
                        with open(Args.namddir+'/RECOMB.%d'%(inicon[i,0]),'w') as f:
                            for j in range(Args.NAMDTIME):
                                f.write('%4d %10.6f\n'%(j+1,pop_rb[i,j]))
                    if Args.LSH != 'FSSH':
                        with open(Args.namddir+'/PSICT.%d'%(inicon[i,0]),'w') as f:
                            for j in range(Args.NAMDTIME):
                                f.write('%4d %13.8f'%(j+1,e_psi[i,j]))
                                for k in range(ibands):
                                    f.write('%10.6f'%(pop_psi[i,j,k]))
                                f.write('\n')


SurfHop()
