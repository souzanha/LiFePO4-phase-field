# -----------------------------------------------------------------------------
# Copyright (c) 2022 Nana Ofori-Opoku. All rights reserved.

import dolfin as df

mpiComm = df.MPI.comm_world
mpiRank = mpiComm.Get_rank()
numproc = mpiComm.Get_size()
isHead = mpiRank == 0

if isHead:
    # df.set_log_active(False)
    # df.set_log_level(df.LogLevel.WARNING)
    # df.set_log_level(df.LogLevel.PROGRESS)
    # df.set_log_level(df.LogLevel.DEBUG)
    df.set_log_level(df.LogLevel.ERROR)
    # df.set_log_level(df.LogLevel.TRACE)
    pass


if not isHead:
    df.set_log_level(df.LogLevel.ERROR)
