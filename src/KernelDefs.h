#ifndef KernelDefs
#define KernelDefs
#include <CL/cl.h>
#include "Kernels/KernelErrors.h"
/*TODO: Determine this number, used for reductions */
#define MAXGROUPS 31

enum KERNEL {
    UpdateZKernel,
    GetNumFromPopsKernel,
    UpdatePKernel,
    mapReduceLogDiffsKernel,
    RDirichletSampleKernel,
    MetroAcceptTestKernel,
    GetNumLociPopsKernel,
    UpdQDirichletKernel,
    FillArrayWRandomKernel,
    InitRandGenKernel,
    UpdateFstKernel,
    FstNormals,
    NumberOfKernels
};


enum BUFFER {
    QCL,
    PCL,
    ZCL,
    GENOCL,
    PREGENOCL,
    RECESSIVECL,
    NUMALLELESCL,
    LAMBDACL,
    POPFLAGCL,
    NUMAFROMPOPSCL,
    NUMLOCIPOPSCL,
    EPSILONCL,
    FSTCL,
    NORMSCL,
    /*LOGTERMSCL,*/
    RANDGENSCL,
    LOGDIFFSCL,
    REDUCERESULTSCL,
    TESTQCL,
    ALPHACL,
    RANDCL, /* buffer for random numbers */
    ERRORCL, /* buffer for error codes */
    NumberOfBuffers
};
typedef struct CLDict {
    cl_kernel *kernels;
    cl_mem *buffers;
    cl_program program;
    size_t *locals;
    cl_platform_id platform_id;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_context context;
    cl_device_id device_id;
    cl_command_queue commands;
} CLDict;
#endif

