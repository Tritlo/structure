/* #pragma OPENCL EXTENSION cl_khr_fp64 : enable */
/* #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable */

//These are inserted during PreProcessingDuringCompilation.

#define GenPos(ind,line,loc) ((ind)*(LINES)*(NUMLOCI)+(line)*(NUMLOCI)+(loc))
#define ZPos(ind,line,loc) ((ind)*(LINES)*(NUMLOCI)+(line)*(NUMLOCI)+(loc))
#define PPos(loc,pop,allele) ((loc)*(MAXPOPS)*(MAXALLELES)+(pop)*(MAXALLELES)+(allele))
#define QPos(ind,pop) ((ind)*(MAXPOPS)+(pop))
#define NumAFromPopPos(pop,allele) ((pop)*(MAXALLELES)+(allele))    /* NumAFromPop */
#define EpsPos(loc,allele) ((loc)*(MAXALLELES)+(allele))    /* Epsilon */
#define AncestDistPos(ind,pop,box) ((ind)*(MAXPOPS)*(NUMBOXES)+(pop)*(NUMBOXES)+(box))

#define UNDERFLO 10e-40f
#define SQUNDERFLO 10e-20f
#define DELTA 10e-5f

/* #include "/home/structure/structure/src/Kernels/util.cl" */
/* #include "/home/structure/structure/src/Kernels/KernelErrors.h" */
/* #include "/home/structure/structure/src/Kernels/UpdateZ.cl" */
/* #include "/home/structure/structure/src/Kernels/UpdateP.cl" */
/* #include "/home/structure/structure/src/Kernels/UpdateQ.cl" */
/* #include "/home/structure/structure/src/Kernels/UpdateFst.cl" */
/* #include "/home/structure/structure/src/Kernels/UpdateAlpha.cl" */
/* #include "/home/structure/structure/src/Kernels/UpdateEpsilon.cl" */
/* #include "/home/structure/structure/src/Kernels/DataCollect.cl" */

#include "util.cl"
#include "KernelErrors.h"
#include "UpdateZ.cl"
#include "UpdateP.cl"
#include "UpdateQ.cl"
#include "UpdateFst.cl"
#include "UpdateAlpha.cl"
#include "UpdateEpsilon.cl"
#include "DataCollect.cl"
