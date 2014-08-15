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

#include "Kernels/util.cl"
#include "Kernels/KernelErrors.h"
#include "Kernels/UpdateZ.cl"
#include "Kernels/UpdateP.cl"
#include "Kernels/UpdateQ.cl"
#include "Kernels/UpdateFst.cl"
#include "Kernels/UpdateAlpha.cl"
#include "Kernels/UpdateEpsilon.cl"
#include "Kernels/DataCollect.cl"
