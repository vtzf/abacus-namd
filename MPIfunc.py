import numpy as np
from mpi4py import MPI
import Args

def MPIfuncE(
    step,nrecv,recvtype,
    savefunc,savename,task,calfunc,e_occ_p,nidx,idx_range
):
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    nprocs = comm.Get_size()

    MPIRTYPE = MPI.DOUBLE
    if recvtype == 'int32':
        MPIRTYPE = MPI.INT
    elif recvtype == 'float32':
        MPISTYPE = MPI.FLOAT

    recvlen = np.cumprod(nrecv)[-1]

    start = MPI.Wtime()
    data_recv_p = np.empty([nidx[myid]]+nrecv,dtype=recvtype)

    for i in range(nidx[myid]):
        data_recv_p[i] = calfunc(i,e_occ_p)

    if myid == 0:
        data_recv = np.empty([step]+nrecv,dtype=recvtype)
    else:
        data_recv = None
    
    comm.Gatherv(data_recv_p,[data_recv,nidx*recvlen,idx_range[0:-1]*recvlen,None],root=0)

    if myid == 0:
        savefunc(savename,data_recv)
        end = MPI.Wtime()
        print('%s time: %.4fs'%(task,end-start))

    comm.Barrier()


def MPIfuncNAC(
    step,nsend,sendtype,nrecv,recvtype,
    loadfunc,savefunc,savename,task,calfunc,
    vec,olp,Lbound,nidx,idx_range
):
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    nprocs = comm.Get_size()

    MPISTYPE = MPI.DOUBLE
    MPIRTYPE = MPI.DOUBLE
    if sendtype == 'int32':
        MPISTYPE = MPI.INT
    elif sendtype == 'float32':
        MPISTYPE = MPI.FLOAT
    if recvtype == 'int32':
        MPIRTYPE = MPI.INT
    elif recvtype == 'float32':
        MPISTYPE = MPI.FLOAT

    sendlen = np.cumprod(nsend)[-1]
    recvlen = np.cumprod(nrecv)[-1]
    nidx_p = np.zeros((nprocs),dtype=int)
    nidx_p[:] = nidx
    nidx_p[-1] -= Args.nstep-step

    start = MPI.Wtime()
    if myid == 0:
        data_send = loadfunc(step,nsend,sendtype)
    else:
        data_send = None

    data_recv_p = np.empty([nidx_p[myid]]+nrecv,dtype=recvtype)
    data_send_p = np.empty([nidx_p[myid]]+nsend,dtype=sendtype)
    comm.Scatterv([data_send,nidx_p*sendlen,idx_range[0:-1]*sendlen,None],data_send_p,root=0)
    if Lbound == 'L':
        vec_p = np.empty([nidx[myid]+1,Args.norbital,nrecv[0]],dtype=float)
        olp_p = np.empty([nidx[myid]+1,Args.norbital,Args.norbital],dtype=float)
        vec_p[1:] = vec
        olp_p[1:] = olp
        mpi_req = []
        if myid != 0:
            mpi_req.append(comm.Irecv(vec_p[0],source=myid-1,tag=0))
            mpi_req.append(comm.Irecv(olp_p[0],source=myid-1,tag=1))
        if myid != nprocs-1:
            mpi_req.append(comm.Isend(vec_p[-1],dest=myid+1,tag=0))
            mpi_req.append(comm.Isend(olp_p[-1],dest=myid+1,tag=1)) 
        MPI.Request.Waitall(mpi_req)
    elif Lbound == 'R':
        vec_p = np.empty([nidx[myid]+1,Args.norbital,nrecv[0]],dtype=float)
        olp_p = np.empty([nidx[myid]+1,Args.norbital,Args.norbital],dtype=float)
        vec_p[0:-1] = vec
        olp_p[0:-1] = olp
        mpi_req = []
        if myid != 0:
            mpi_req.append(comm.Isend(vec_p[0],dest=myid-1,tag=0))
            mpi_req.append(comm.Isend(olp_p[0],dest=myid-1,tag=1))
        if myid != nprocs-1:
            mpi_req.append(comm.Irecv(vec_p[-1],source=myid+1,tag=0))
            mpi_req.append(comm.Irecv(olp_p[-1],source=myid+1,tag=1))
        MPI.Request.Waitall(mpi_req)
    else:
        vec_p = vec
        olp_p = olp
        
    for i in range(nidx_p[myid]):
        data_recv_p[i] = calfunc(myid,i,data_send_p,vec_p,olp_p)

    if myid == 0:
        data_recv = np.empty([step]+nrecv,dtype=recvtype)
    else:
        data_recv = None
    
    comm.Gatherv(data_recv_p,[data_recv,nidx_p*recvlen,idx_range[0:-1]*recvlen,None],root=0)

    if myid == 0:
        savefunc(savename,data_recv)
        end = MPI.Wtime()
        print('%s time: %.4fs'%(task,end-start))

    comm.Barrier()
