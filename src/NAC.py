#!/usr/bin/env python
#-*- encoding:utf-8 -*-
import numpy as np
import Args
import NACfunc
import MPIfunc

def NACcalc():
    myid = Args.comm.Get_rank()
    nprocs = Args.comm.Get_size()

    idx_max = (Args.nstep*(np.arange(nprocs)+1))//nprocs
    idx_min = (Args.nstep*np.arange(nprocs))//nprocs
    nidx = idx_max - idx_min
    idx_range = np.zeros((nprocs+1),dtype=int)
    idx_range[1:] = np.cumsum(nidx)

    start = MPIfunc.MPI.Wtime()
    e_occ_p, vec_p = NACfunc.LoadDataOrbital(
                     'float64',myid,nprocs,nidx,idx_range)
    olp_p, olps_p = NACfunc.LoadDataOlp(
                    'float64',myid,nprocs,nidx,idx_range,Args.LSYNS)
    end = MPIfunc.MPI.Wtime()
    if myid == nprocs-1:
        print('Reading time: %.4fs'%(end-start))
    Args.comm.Barrier()

#    if myid == 0:
#        np.save('idx_range.npy',idx_range)
#    np.save('e_occ-%d.npy'%myid,e_occ_p)
#    np.save('vec-%d.npy'%myid,vec_p)
#    np.save('olp-%d.npy'%myid,olp_p)
#    np.save('olps-%d.npy'%myid,olps_p)

    if Args.LRANGE:
        MPIfunc.MPIfuncE(
            Args.nstep,[Args.nbands+2],'float64',NACfunc.SaveE,
            'all_en.npy','Energy and INICON',NACfunc.ReadE1,
            e_occ_p[int(bool(myid)):],nidx,idx_range
        )
    else:
        MPIfunc.MPIfuncE(
            Args.nstep,[Args.nbands+2],'float64',NACfunc.SaveE,
            'all_en.npy','Energy',NACfunc.ReadE,
            e_occ_p[int(bool(myid)):],nidx,idx_range
        )
    bandrange = open(Args.namddir+'bandrange.dat').readline().split()
    Args.iband_s = int(bandrange[0])
    Args.iband_e = int(bandrange[1])
    Args.ibands = Args.iband_e-Args.iband_s+1
    vec_s = np.ascontiguousarray(vec_p[:,:,Args.iband_s-1:Args.iband_e])
    Args.comm.Barrier()

    if Args.LCHARGE:
        MPIfunc.MPIfuncNAC(
            Args.nstep,[0],'int32',[Args.nbands],'float64',NACfunc.LoadDataNull,
            NACfunc.SaveCD,'all_wht.npy','Charge density',NACfunc.CDInfo,
            vec_p,olp_p,olps_p,'N',nidx,idx_range
        )
    if Args.LPHASE:
        MPIfunc.MPIfuncNAC(
            Args.nstep,[0],'int32',[Args.ibands],'int32',NACfunc.LoadDataNull,
            NACfunc.SavePhase,'phase_m.npy','Phase correction',NACfunc.PhaseInfo,
            vec_s,olp_p,olps_p,'L',nidx,idx_range
        )
        if Args.LRANGE:
            MPIfunc.MPIfuncNAC(
                Args.nstep-1,[Args.ibands,Args.ibands],'int32',
                [Args.ibands,Args.ibands],'float64',NACfunc.LoadDataPhase,
                NACfunc.SaveNAC,'NATXT','NAC with phase correction',NACfunc.NACPhaseInfo,
                vec_s,olp_p,olps_p,'R',nidx,idx_range
            )
        else:
            MPIfunc.MPIfuncNAC(
                Args.nstep-1,[Args.ibands,Args.ibands],'int32',
                [Args.ibands,Args.ibands],'float64',NACfunc.LoadDataPhase,
                NACfunc.SaveCOUP,'COUPCAR','NAC with phase correction',NACfunc.NACPhaseInfo,
                vec_s,olp_p,olps_p,'R',nidx,idx_range
            )
    else:
        if Args.LRANGE:
            MPIfunc.MPIfuncNAC(
                Args.nstep-1,[0],'int32',[Args.ibands,Args.ibands],'float64',
                NACfunc.LoadDataNull,NACfunc.SaveNAC,
                'NATXT','NAC without phase correction',NACfunc.NACInfo,
                vec_s,olp_p,olps_p,'R',nidx,idx_range
            )
        else:
            MPIfunc.MPIfuncNAC(
                Args.nstep-1,[0],'int32',[Args.ibands,Args.ibands],'float64',
                NACfunc.LoadDataNull,NACfunc.SaveCOUP,
                'COUPCAR','NAC without phase correction',NACfunc.NACInfo,
                vec_s,olp_p,olps_p,'R',nidx,idx_range
            )


if __name__ == "__main__":
    NACcalc()
