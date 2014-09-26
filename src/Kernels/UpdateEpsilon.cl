


__kernel void NonIndUpdateEpsilon(
        __global float *P,
        __global float *Epsilon,
        __global float *Fst,
        __global int *NumAlleles,
        __global uint *randGens,
        __global float *lambdas,
        const float invsqrtnuminds)
{

    int loc = get_global_id(0);
    int allele1,allele2;
    int eps1,eps2;
    int pop;
    float diff;
    float sum;
    float lambda = lambdas[0];
    float rand;
    float c, y,t;
    float frac;

    RndDiscState randState[1];
    initRndDiscState(randState,randGens,loc);
    
    while (loc < NUMLOCI){
        int numalls = NumAlleles[loc];
        if (numalls > 1){

            allele1 = RandomInteger(0,numalls,randState);
            /* get second allele != allele1 */
            allele2 = RandomInteger(0,numalls-1,randState);
            if (allele2 >= allele1)
            {
                allele2 += 1;
            }

            eps1 = Epsilon[EpsPos(loc,allele1)];
            eps2 = Epsilon[EpsPos(loc,allele2)];
            diff = invsqrtnuminds*rndDisc(randState);
            /* diff = numToRange(0,invsqrtnuminds,rndDisc(randState)); */

            if( (eps1 + diff < 1.0f) && (eps2 - diff > 0.0f)){
                //TODO: Evaluate whether we should reduce here.
                sum=0.0f;
                c = 0.0f;
                for (pop=0; pop<MAXPOPS; pop++) { /*compute likelihood ratio*/
                    frac = (1.0f-Fst[pop])/Fst[pop];

                    y =  lgamma(frac*eps1) -c;
                    t = sum +y;
                    c = (t - sum ) - y;
                    sum = t;

                    y =  lgamma(frac*eps2) -c;
                    t = sum +y;
                    c = (t - sum ) - y;
                    sum = t;

                    y = -lgamma(frac*(eps1+diff)) - c;
                    t = sum +y;
                    c = (t - sum ) - y;
                    sum = t;

                    y = -lgamma(frac*(eps2-diff)) - c;
                    t = sum +y;
                    c = (t - sum ) - y;
                    sum = t;


                    y = frac*diff*log(P[PPos(loc, pop, allele1)]) - c;
                    t = sum +y;
                    c = (t - sum ) - y;
                    sum = t;

                    y = -frac*diff*log(P[PPos(loc, pop, allele2)]) - c;
                    t = sum +y;
                    c = (t - sum ) - y;
                    sum = t;

                }
                /* if (fabs(lambda - 1.0f) > 10e-10) {              /1* compute prior ratio *1/ */
                /*     /1* as it is in code *1/ */
                /*     /1* float ratio = (eps1 + diff)* (eps2 - diff)/(eps1)/(eps2) *1/ */
                /*     /1* sum += log(pow(ratio, lambda-1.0f)); *1/ */
                /*     /1* as it probably should be ? *1/ */
                /*     float ratio = (eps1 + diff)* (eps2 - diff)/(eps1*eps2); */
                /*     sum += (lambda-1.0f)*log(ratio); */
                /* } */
                float randVal = rndDisc(randState);
                if (randVal < exp(sum) || log(randVal) < sum ){
                    AtomicAdd(&Epsilon[EpsPos(loc,allele1)],diff);
                    AtomicAdd(&Epsilon[EpsPos(loc,allele2)],-diff);
                }
            }
        }
        loc += get_global_size(0);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    saveRndDiscState(randState);
}
