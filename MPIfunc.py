import numpy as np
import time
from mpi4py import MPI
import Args

def MPIfunc(
    step,nsend,sendtype,nrecv,recvtype,
    loadfunc,savefunc,savename,task,calfunc
):
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    myid = comm.Get_rank()
    nprocs = comm.Get_size()
    master = 0

    MPISTYPE = MPI.DOUBLE
    MPIRTYPE = MPI.DOUBLE
    if sendtype == 'int32':
        MPISTYPE = MPI.INT
    if recvtype == 'int32':
        MPIRTYPE = MPI.INT

    if master == myid:
        status = MPI.Status()
        data_send = loadfunc(step,nsend,sendtype)
        data_recv = np.empty(nrecv,dtype=recvtype)
        data = np.zeros([step]+nrecv,dtype=recvtype)
        null = np.zeros((0),dtype=sendtype)
        numsent = 0

        start = time.time()
        if nprocs-1 < step:
            for i in range(1,nprocs):
                comm.Send([data_send[i-1],None,MPISTYPE],dest=i,tag=i)
                numsent += 1
            for i in range(1,step+1):
                status = MPI.Status()
                comm.Recv([data_recv,None,MPIRTYPE],\
                           source=MPI.ANY_SOURCE,\
                           tag=MPI.ANY_TAG,status=status)
                sender = status.Get_source()
                anstype = status.Get_tag()
                data[anstype-1] = data_recv
                if numsent < step:
                    comm.Send([data_send[numsent],None,MPISTYPE],\
                               dest=sender,tag=numsent+1)
                    numsent += 1
                else:
                    comm.Send([null,None,MPISTYPE],dest=sender,tag=0)
        else:
            for i in range(1,step+1):
                comm.Send([data_send[i-1],None,MPISTYPE],dest=i,tag=i)
            for i in range(step+1,nprocs):
                comm.Send([null,None,MPISTYPE],dest=i,tag=0)
            for i in range(1,step+1):
                status = MPI.Status()
                comm.Recv([data_recv,None,MPIRTYPE],\
                           source=MPI.ANY_SOURCE,\
                           tag=MPI.ANY_TAG,status=status)
                sender = status.Get_source()
                anstype = status.Get_tag()
                data[anstype-1] = data_recv
                comm.Send([null,None,MPISTYPE],dest=sender,tag=0)

        savefunc(savename,data)
        end = time.time()
        print('%s time: %.4fs'%(task,end-start))

    else:
        while True:
            data_recv = np.empty(nsend,dtype=sendtype)
            status = MPI.Status()
            comm.Recv([data_recv,None,MPISTYPE],\
                       source=MPI.ANY_SOURCE,\
                       tag=MPI.ANY_TAG,status=status)
            num = status.Get_tag()
            if num != 0:
                data_send = calfunc(num+Args.start_t-1,data_recv)
                comm.Send([data_send,None,MPIRTYPE],dest=master,tag=num)
            else:
                break

