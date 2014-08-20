float
FPriorDiff (float newf, float oldf)
{
    /*returns log diff in priors for the correlation, f. See notes 5/14/99, and 7/15/99 */
    float pri = (FPRIORMEAN*FPRIORMEAN/(FPRIORSD*FPRIORSD) - 1);

    return (pri * (log (newf) - log( oldf)) + (oldf - newf) *FPRIORMEAN/(FPRIORSD*FPRIORSD));
}



float FlikeFreqsDiffMap (float newfrac,float oldfrac,
        __global float *Epsilon,
        __global float *P,
        __global int *NumAlleles,
        int loc,int pop){
    int allele;
    float eps,logp;
    float sum;
    float c, y,t;

    if (NumAlleles[loc]==0) {
        return (lgamma(oldfrac) + lgamma(newfrac)); /* should not be counting sites with all missing data */
    } else {
        sum = 0.0f;
        c = 0.0f;
        for (allele=0; allele < NumAlleles[loc]; allele++) {
            eps = Epsilon[EpsPos (loc, allele)];
            logp = log(P[PPos(loc,pop,allele)]);

            /* sum += newfrac*eps*logp; */
            y =  newfrac*eps*logp - c;
            t = sum + y;
            c = (t-sum) -y;
            sum = t;

            /* sum -= oldfrac*eps*logp; */
            y = -oldfrac*eps*logp - c;
            t = sum + y;
            c = (t-sum) -y;
            sum = t;

            /* sum += lgamma( oldfrac*eps); */
            y = lgamma( oldfrac*eps) - c;
            t = sum + y;
            c = (t-sum) -y;
            sum = t;

            /* sum -= lgamma( newfrac*eps); */
            y = -lgamma( newfrac*eps) - c;
            t = sum + y;
            c = (t-sum) -y;
            sum = t;
        }
        return sum;
    }
}



__kernel void UpdateFst(
            __global float *Epsilon,
            __global float *Fst,
            __global float *P,
            __global int *NumAlleles,
            __global float *normals,
            __global uint *randGens,
            __global float *results,
            __local  float *scratch,
            const int numfst)
{
    int pop = get_global_id(1);
    int numgroups = get_num_groups(0);
    float c, y, t;// KahanSum
    while (pop < numfst){
        int loc = get_global_id(0);
        float newf = normals[pop];
        /* ensure newf is large enough so we don't cause over/underflow */
        if (newf > 10e-5f && newf < 1.0f){
            float sum = 0.0f;
            int redpop;
            int numredpops;
            float oldf = Fst[pop];
            float newfrac = (1.0f-newf)/newf;
            float oldfrac = (1.0f-oldf)/oldf;
            numredpops = pop +1;
            if (ONEFST) numredpops = MAXPOPS;
            /* idempotent */
            /* Map and partial reduce */
            c = 0.0f;
            while( loc < NUMLOCI){
                float elem = 0.0f;
                float d = 0.0f;
                float z,u;
                for(redpop = pop; redpop < numredpops; redpop++){
                    z = FlikeFreqsDiffMap(newfrac,oldfrac,Epsilon,P,NumAlleles,loc,redpop) - d;
                    u = elem + z;
                    d = (u - elem) -z;
                    elem = u;
                }
                /* Kahan summation */
                y = elem - c;
                t = sum + y;
                c = (t - sum) -y;
                sum = t;
                loc += get_global_size(0);
            }
            /* reduce locally */
            int localLoc = get_local_id(0);
            scratch[localLoc] = sum;
            barrier(CLK_LOCAL_MEM_FENCE);
            int devs = get_local_size(0);
            c = 0.0f;
            for(int offset = get_local_size(0) /2; offset > 0; offset >>= 1){
                if(localLoc < offset){
                     y = scratch[localLoc + offset] -c;
                     t = scratch[localLoc] + y;
                     c = (t-scratch[localLoc]) -y;
                     scratch[localLoc] = t;

                }
                //Handle if were not working on a multiple of 2
                if (localLoc == 0 && (devs-1)/2 == offset){
                     y = scratch[devs-1] -c;
                     t = scratch[localLoc] + y;
                     c = (t-scratch[localLoc]) -y;
                     scratch[localLoc] = t;
                }
                devs >>= 1;
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            int gid = get_group_id(0);
            //TODO: Handle if numgroups are more than MAXGROUPS
            //Possibly by reducing with global barrier.
            if(localLoc == 0){
                results[pop*numgroups +gid] = scratch[0];
            }

            barrier(CLK_GLOBAL_MEM_FENCE);
            if(gid==0){
                RndDiscState randState[1];
                initRndDiscState(randState,randGens,pop);
                int multiple = 1;
                if (ONEFST) multiple = MAXPOPS;
                float logprobdiff = FPriorDiff (newf, oldf);
                logprobdiff += multiple*NUMLOCI*lgamma(newfrac);
                logprobdiff -= multiple*NUMLOCI*lgamma(oldfrac);
                for(int id =0; id < numgroups; id ++){
                    logprobdiff += results[pop*numgroups + id];
                    results[pop*numgroups + id] = 0;
                }
                if (logprobdiff >= 0.0f || rndDisc(randState) < exp(logprobdiff)) {   /*accept new f */
                    for(redpop = pop; redpop < numredpops; redpop++){
                        Fst[redpop] = newf;
                    }
                }
                saveRndDiscState(randState);
            }
        }
        pop += get_global_size(1);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

