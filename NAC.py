#!/usr/bin/env python
#-*- encoding:utf-8 -*-
import Args
import NACfunc
import MPIfunc

def NACcalc():
    if Args.LRANGE:
        MPIfunc.MPIfunc_eq(
            Args.nstep,[0],'int32',[Args.nbands+2],'float64',
            NACfunc.LoadDataNull,NACfunc.SaveE1,
            'all_en.npy','Energy and INICON',NACfunc.ReadE1
        )
        Args.comm.Barrier()
        bandrange = open(Args.namddir+'bandrange.dat').readline().split()
        Args.iband_s = int(bandrange[0])
        Args.iband_e = int(bandrange[1])
        Args.ibands = Args.iband_e-Args.iband_s+1
    else:
        MPIfunc.MPIfunc_eq(
            Args.nstep,[0],'int32',[Args.nbands],'float64',
            NACfunc.LoadDataNull,NACfunc.SaveE,
            'all_en.npy','Energy',NACfunc.ReadE
        )

    if Args.LCHARGE:
        MPIfunc.MPIfunc_eq(
            Args.nstep,[0],'int32',[Args.nbands],'float64',NACfunc.LoadDataNull,
            NACfunc.SaveCD,'all_wht.npy','Charge density',NACfunc.CDInfo
        )

    if Args.LPHASE:
        MPIfunc.MPIfunc_eq(
            Args.nstep,[0],'int32',[Args.ibands],'int32',NACfunc.LoadDataNull,
            NACfunc.SavePhase,'phase_m.npy','Phase correction',NACfunc.PhaseInfo
        )
        if Args.LRANGE:
            MPIfunc.MPIfunc_eq(
                Args.nstep-1,[Args.ibands,Args.ibands],'int32',
                [Args.ibands,Args.ibands],'float64',NACfunc.LoadDataPhase,
                NACfunc.SaveNAC,'NATXT','NAC with phase correction',NACfunc.NACPhaseInfo
            )
        else:
            MPIfunc.MPIfunc_eq(
                Args.nstep-1,[Args.ibands,Args.ibands],'int32',
                [Args.ibands,Args.ibands],'float64',NACfunc.LoadDataPhase,
                NACfunc.SaveCOUP,'COUPCAR','NAC with phase correction',NACfunc.NACPhaseInfo
            )
    else:
        if Args.LRANGE:
            MPIfunc.MPIfunc_eq(
                Args.nstep-1,[0],'int32',[Args.ibands,Args.ibands],'float64',
                NACfunc.LoadDataNull,NACfunc.SaveNAC,
                'NATXT','NAC without phase correction',NACfunc.NACInfo
            )
        else:
            MPIfunc.MPIfunc_eq(
                Args.nstep-1,[0],'int32',[Args.ibands,Args.ibands],'float64',
                NACfunc.LoadDataNull,NACfunc.SaveCOUP,
                'COUPCAR','NAC without phase correction',NACfunc.NACInfo
            )


if __name__ == "__main__":
    NACcalc()
