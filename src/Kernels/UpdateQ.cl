#include "Kernels/calcLike.cl"

__kernel void MetroAcceptTest(
        __global float *TestQ,
        __global float *Q,
        __global uint *randGens,
        __global float *logdiffs,
        __global int *popflags)
{
    int ind = get_global_id(0);
    int pop;
    RndDiscState randState[1];
    
    while (ind < NUMINDS){
        initRndDiscState(randState,randGens,ind);
        if (!((USEPOPINFO) && (popflags[ind]))) {
            if(rndDisc(randState) < exp(logdiffs[ind])){
                for (pop = 0; pop < MAXPOPS; pop++) {
                    Q[QPos (ind, pop)] = TestQ[QPos(ind,pop)];
                }
            }
        }
        saveRndDiscState(randState);
        ind += get_global_size(0);
    }
}

__kernel void GetNumLociPops(
        __global int *Z,
        __global int *popflags,
        __global int *NumLociPops)
{
    int ind = get_global_id(0);

    while(ind < NUMINDS){
        int loc = get_global_id(1);
        int offset = ind*MAXPOPS;
        int line, from,pop;
        while( loc < NUMLOCI){
            /* initialize the NumLociPops array */
            /* if(get_global_id(1) == 0){ */
            /*     for(pop = 0; pop < MAXPOPS; pop++){ */
            /*         NumLociPops[pop+offset] = 0; */
            /*     } */
            /* } */
            /* barrier(CLK_GLOBAL_MEM_FENCE); */
            if(ind < NUMINDS && loc < NUMLOCI) {
                if (!((USEPOPINFO) && (popflags[ind]))) {
                    for (line = 0; line < LINES; line++) {
                        from = Z[ZPos (ind, line, loc)];
                        if (from != UNASSIGNED) {
                            AtomicInc(&NumLociPops[from+offset]);
                        }
                    }
                }
            }
            loc += get_global_size(1);
        }
        ind += get_global_size(0);
    }
}

__kernel void UpdQDirichlet(
        __global float *Alpha,
        __global int *NumLociPops,
        __global uint *randGens,
        __global float *Q,
        __global int *popflags)
{
    int ind = get_global_id(0);
    RndDiscState randState[1];
    //TODO: Add PopFlag here
    if (!((USEPOPINFO) && (popflags[ind]))) {
        while (ind < NUMINDS){
            initRndDiscState(randState,randGens,ind);
            float GammaSample[MAXPOPS];

            int i = 0;
            float sum = 0.0f;
            int offset = ind*MAXPOPS;
            float param;
            for(i = 0; i < MAXPOPS; i++){
                param = Alpha[i]+NumLociPops[i+offset];
                GammaSample[i] = RGammaDisc(param,1,randState);
                sum += GammaSample[i];
            }
            for(i = 0; i < MAXPOPS; i++){
                Q[i+offset] = GammaSample[i]/sum;
            }
            saveRndDiscState(randState);
            ind += get_global_size(0);
        }
    }
}


