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


/* #include "mwc64x.cl" */

#ifndef RANDGEN
#define RANDGEN
/* #define MAXRANDVAL 4294967296 */
/* #define MAXRANDVAL 281474976710656 */
#define MAXRANDVAL 2147483647
#define RANDSPERSTREAM 2147483648
#define PI 3.14159265359f


uint getRandGen(__global uint *randGens, int id){
    return  randGens[id];
}
void saveRandGen(__global uint *randGens, int randGen,uint rng){
    randGens[randGen]=rng;
}

float uintToUnit(uint rnduint){
   return (float) rnduint / MAXRANDVAL;
}


__kernel void InitRandGens( __global uint *randGens, const int baseOffset)
{
    int pos = get_global_id(0);
    /* mwc64x_state_t rng; */
    /* MWC64X_SeedStreams(&rng,baseOffset,RANDSPERSTREAM); */
    /* saveRandGen(randGens,pos,rng); */
}

/*#if DEBUGCOMPARE
typedef struct RndDiscState {
    __global float *randomArr;
    int randomValsTaken;
    int baseOffset;
} RndDiscState;


void initRndDiscState(RndDiscState *state,__global float * randomArr, int offset)
{
    state->randomArr = randomArr;
    state->maxrandom = maxrandom;
    state->baseOffset = offset;
    state->randomValsTaken = 0;
}

float rndDisc(RndDiscState * state)
{
    float val;
    val = state->randomArr[state->baseOffset + state->randomValsTaken];
    state->randomValsTaken++;
    return val;
}
#else*/
typedef struct RndDiscState {
    __global uint *randGens;
    int randGenId;
    uint rng;
} RndDiscState;

void initRndDiscState(RndDiscState *state, __global uint * randGens, int id)
{
    state->randGens = randGens;
    state->randGenId = id;
    state->rng = getRandGen(randGens,id);
}

uint rndUInt(RndDiscState *state){
    /* Drand 48 */
    /* const unsigned int a = 25214903917; */
    /* const unsigned int c = 11; */
    /* const unsigned int m = 281474976710656 -1; */
    /* GGL */
    const uint a = 16807;
    const uint c = 0;
    const uint m = 2147483647;
    uint x = state->rng;
    /* uint xn = (a*x + c) % m; */
    /* uint xn = (a*x + c) & m; */
    uint xn = a*x % m;
    state->rng = xn;
    /* printf("%d, id: %d\n",xn,state->randGenId); */
    /* printf("%d, %d\n",get_global_id(0),get_global_id(1)); */
    return xn;
}

/* uint rndUInt(RndDiscState *state){ */
/* return MWC64X_NextUint(&(state->rng)); */
/* } */

float rndDisc(RndDiscState * state)
{
    /* uint rand = rndUInt(state); */
    /* float val = uintToUnit(rand); */
    /* printf("%f\n",val); */
    return uintToUnit(rndUInt(state));
}

/* uint rndUInt(RndDiscState * state) */
/* { */
/*     return rndUInt(s/ate); */
/* } */

float2 BoxMuller(RndDiscState *state)
{
  float u0=rndDisc(state), u1=rndDisc(state);
  float r=sqrt(-2*log(u0));
  float theta=2*PI*u1;
  return (float2) (r*sin(theta),r*cos(theta));
}

float2 BoxMullerF(RndDiscState *state)
{
  float u0=rndDisc(state), u1=rndDisc(state);
  float r=sqrt(-2*log(u0));
  float theta=2*PI*u1;
  return (float2) (r*sin(theta),r*cos(theta));
}



void saveRndDiscState(RndDiscState *state){
    saveRandGen(state->randGens,state->randGenId,state->rng);
}
/*#endif*/

#endif

#define RAND_MAX 4294967296.0f

inline void AtomicInc(__global int *source){
    atomic_add(source,1);
}

inline void AtomicAdd(volatile __global float *source, const float operand) {
    *source += operand;
    /* union { */
    /*     unsigned int intVal; */
    /*     float floatVal; */
    /* } newVal; */
    /* union { */
    /*     unsigned int intVal; */
    /*     float floatVal; */
    /* } prevVal; */
    /* do { */
    /*     prevVal.floatVal = *source; */
    /*     newVal.floatVal = prevVal.floatVal + operand; */
    /* } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal); */
}
/*
 * returns a random real in the [lower,upper)
 */
float randomReal(float lower, float upper,RndDiscState *randState)
{
    uint randVal;
    float randPercent;
    float random;

    randVal = rndUInt(randState);
    randPercent = (float) randVal/(RAND_MAX +1);
    return (lower + randPercent*(upper-lower));
}

float numToRange(float low, float high, float num)
{
    /* Takes a number in [0,1) -> [low,high) */
    return (low + num * (high - low) );
}

int dimLoc(int * dims, int * dimMaxs, int numDims)
{
    int loc = 0;
    int i, j;
    for(i = 0; i < numDims; ++i) {
        int dimProd =1;
        for(j = i+1; j < numDims; j++) {
            dimProd *= dimMaxs[j];
        }
        loc += dims[i]*dimProd;
    }
    return loc;
}
/*
 * Copies the entire last dimension over into localarr
 */
void copyToLocal( __global float * globalArr, float *localArr,
                  int * dims, int * dimMaxs, int numDims)
{
    int i;
    int origLastDim = dims[numDims-1];
    for(i = 0; i < dimMaxs[numDims-1]; ++i) {
        dims[numDims-1] = i;
        localArr[i] = globalArr[dimLoc(dims,dimMaxs,numDims)];
    }
    dims[numDims-1] = origLastDim;
}


/*
 *  Returns a random number between 0 and n-1, according to a list of
 *  probabilities.  The function takes a (possibly) unnormalized
 *  vector of probabilities of the form (p1, p2, p3,....,p_n).  There
 *  are "total" possible options, and sum is the sum of the
 *  probabilities.  This comes up in the Gibbs sampler context.
 */
int PickAnOptionDiscrete(int total, float sum, float Probs [],
                         RndDiscState *randState)
{
    int option;
    float random;
    float sumsofar =  0.0f;

    random = numToRange(0,sum, rndDisc(randState));
    for (option=0; option < total; ++option) {
        sumsofar += Probs[option];
        if (random <= sumsofar) {
            break;
        }
    }
    return option;
}

/* int RandomInteger(int low, int high,RndDiscState *randState) */
/* { */
/*     if (high == low){ */
/*         return low; */
/*     } */
/*     int range = high-low; */
/*     uint random = rndUInt(randState) % range; */
/*     return (int) random + low; */
/* } */

int RandomInteger(int low, int high,RndDiscState *randState)
{
    int k;
    float d = rndDisc(randState);

    k = (int) (d * (high - low + 1));
    return (low + k);
}


int AlphaPos(int loc, int pop)
{
    if((loc)==NUMLOCATIONS) {
        return pop;
    } else {
        return (MAXPOPS*((loc)+1)+(pop));
    }
}

/* Returns gamma(f,1), where 0 < f < 1 */
float RGammaDiscFloat(float n,RndDiscState *randState){
    float x=0.0f;
    float E=2.71828182f;
    float b=(n+E)/E;
    float p=0.0f;
    /* int counter = 0; */
    while(1){
        p=b*rndDisc(randState);
        if(p>1) {
            x=-log((b-p)/n);
            if (!(((n-1)*log(x))<log(rndDisc(randState)))) {
                /* Accept */
                break;
            }
        } else {
            x=exp(log(p)/n);
            if(!(x>-log(rndDisc(randState)))) {
                /* Accept */
                break;
            }
        }
        /* counter += 1; */
        /* if (counter >= 5000){ */
        /*     printf("To many iterations reached!%d\n",0); */
        /*     return x; */
        /* } */
    }
    return x;
}

/* Returns gamma(1,1) */
float RGammaDiscOne(RndDiscState *randState){
    float a=0.0f;
    float u,u0,ustar;
    u=rndDisc(randState);
    u0=u;
    while (1){
        ustar=rndDisc(randState);
        if(u<ustar) {
            break;
        }
        u=rndDisc(randState);
        if(u>=ustar) {
            a += 1;
            u=rndDisc(randState);
            u0=u;
        }
    }
    return (a+u0);
}

/* Returns gamma(n,1) where n is an int */
float RGammaDiscInt(int n,RndDiscState *randState){
    int i =0;
    float x = 0;
    for(i = 0; i < n; ++i){
        x += log(rndDisc(randState));
    }
    return -x;
}

/*  taken from page 413 of
 *
 *  Non-Uniform Random Variate Generation by Luc Devroye
 *
 *  (originally published with Springer-Verlag, New York, 1986)
 *
 *  which can be found online at http://luc.devroye.org/rnbookindex.html
 */
float RGammaCheng(float a,RndDiscState *randState){
    float b = a - log(4.0f);
    float c = a + sqrt(2.0f*a-1.0f);
    float U,V,X,Y,Z,R;
    while (1){
        U = rndDisc(randState);
        V = rndDisc(randState);
        Y = a*log(V/(1.0f-V));
        X = a*exp(V);
        Z = U*(V*V);
        R = b + c*Y - X;
        if( (R >= (9.0f/2.0f)*Z - (1+log(9.0f/2.0f))) ||   ( R >= log(Z)) ){
            break;
        }
    }
    return X;
}

/*  taken from page 410 of
 *
 *  Non-Uniform Random Variate Generation by Luc Devroye
 *
 *  (originally published with Springer-Verlag, New York, 1986)
 *
 *  which can be found online at http://luc.devroye.org/rnbookindex.html
 */

float RGammaBest(float a,RndDiscState *randState){
    float b = a -1;
    float c = 3*a - 0.75f;
    float U,V,W,Y,X,Z;
    while (1){
        U = rndDisc(randState);
        V = rndDisc(randState);
        W = U*(1-U);
        Y = sqrt((c/W))*(U-0.5f);
        X = b + Y;
        if( X >= 0){
            Z = 64*(W*W*W)*(V*V);
            if((Z <= 1 - (2*(Y*Y))/X) || (log(Z) <= 2*(b*log(X/b)-Y))){
                break;
            }
        }
    }
    return X;
}

float RGammaLargerThanOne(float n, RndDiscState *randState)
{
    float aa,w,x;
    float nprev=0.0f;
    float c1=0.0f;
    float c2=0.0f;
    float c3=0.0f;
    float c4=0.0f;
    float c5=0.0f;
    float u1;
    float u2;
    /*if(n!=nprev) {*/
        c1=n-1.0f;
        aa=1.0f/c1;
        c2=aa*(n-1/(6*n));
        c3=2*aa;
        c4=c3+2;
        if(n>2.5f) {
            c5=1/sqrt(n);
        }
    /*}*/
four:
    u1=rndDisc(randState);
    u2=rndDisc(randState);
    if(n<=2.5f) {
        goto five;
    }
    u1=u2+c5*(1-1.86f*u1);
    if ((u1<=0) || (u1>=1)) {
        goto four;
    }
five:
    w=c2*u2/u1;
    if(c3*u1+w+1.0f/w < c4) {
        goto six;
    }
    if(c3*log(u1)-log(w)+w >=1) {
        goto four;
    }
six:
    x=c1*w;
    nprev=n;
    return x;

}

/*-----------Gamma and dirichlet from Matt.----------*/
/* gamma random generator from Ripley, 1987, P230 */

float RGammaDisc(float n,float lambda,RndDiscState *randState)
{
    float x=0.0f;
    if(n<1) {
        x = RGammaDiscFloat(n,randState);
    } else if(n==1.0f) {
        /* gamma (1,1) is an exponential dist */
        /*x = RGammaDiscOne(randState);*/
        x = -log(rndDisc(randState));
    } else {
        /*if (rndDisc(randState) < 0.5){*/
        x = RGammaBest(n,randState);
        /* x = RGammaLargerThanOne(n,randState); */
        /* x = RGammaCheng(n,randState); */
        /*} else {*/
            /*x = RGammaCheng(n,randState);*/
        /*}*/
        /*int wholepart = (int) n;*/
        /*float xi = 0.0;*/
        /*float wholegamma = 0.0;*/
        /*float floatpart = n - wholepart;*/
        /*xi = RGammaDiscFloat(floatpart,randState);*/
        /*wholegamma = RGammaBest(wholepart,randState);*/
        /*x = xi + wholegamma;*/
    }
    return x/lambda;
}





/* Dirichlet random generator
   a and b are arrays of length k, containing floats.
   a is the array of parameters
   b is the output array, where b ~ Dirichlet(a)
   */

void RDirichletDisc(float * a, int k, float * b,RndDiscState *randState)
{
    int i;
    float sum=0.0f;
    float c,y,t;
    c = 0.0f;
    for(i=0; i<k; i++) {
        b[i]=RGammaDisc(a[i],1,randState);
        y = b[i] - c;
        t = sum + y;
        c = (t - sum) -y;
        sum = t;
    }
    for(i=0; i<k; i++) {
        b[i] /= sum;
    }
}

/* Melissa's version, adapted from an algorithm on wikipedia.  January 08 */
float LogRGammaDisc(float n, float lambda, RndDiscState *randState)
{
    float v0, v[3], E=2.71828182f, em, logem, lognm;
    int i;
    if (n >= 1.0f) {
        return log(RGammaDisc(n, lambda,randState));
    }
    v0 = E/(E+n);
    while (1) {
        for (i=0; i<3; i++) {
            v[i] = rndDisc(randState);
        }

        if (v[0] <= v0) {
            logem = 1.0f/n*log(v[1]);
            em = exp(logem);
            lognm = log(v[2])+(n-1)*logem;
        } else {
            em = 1.0f-log(v[1]);
            logem = log(em);
            lognm = log(v[2]) - em;
        }
        if (lognm <= (n-1)*logem - em) {
            return logem - log(lambda);
        }
    }
}

/*O(k)*/
void LogRDirichletDisc (float *a, int k,__global float *b,
                        RndDiscState *randState)
{
    int i;
    float sum = 0.0f;
    /* float sum2; */
    float c,y,t;
    c = 0.0f;
    for (i = 0; i < k; i++) {
        b[i] =RGammaDisc (a[i], 1,randState);
        y = b[i] - c;
        t = sum + y;
        c = (t - sum) -y;
        sum = t;
    }

    /* patch added May 2007 to set gene frequencies equal if all draws
       from the Gamma distribution are very low.
       Ensures that P and logP remain defined in this rare event */
    /* if(sum<UNDERFLO) { */
    /*     for(i=0; i<k; i++) { */
    /*         b[i] = 1.0/(float)(k); */
    /*     } */
    /* } else { */
        /* sum2=log(sum); */
        for (i = 0; i < k; i++) {
            b[i]/=sum;
        }
    /* } */
}


enum KERNELERROR {
    KERNEL_SUCCESS,
    KERNEL_OUT_OF_BOUNDS,
    NumberOfKernelErros
};



__kernel void Dirichlet(
        __global float *Parameters,
        __global uint *randGens,
        __global float *TestQ)
{
    /* printf("Kernel: Dirichlet\n"); */
    int ind = get_global_id(0);
    RndDiscState randState[1];

    while (ind < NUMINDS){
        initRndDiscState(randState,randGens,ind);
        float GammaSample[MAXPOPS];

        int i = 0;
        float sum = 0.0f;
        int offset = ind*MAXPOPS;
        for(i = 0; i < MAXPOPS; i++){
            GammaSample[i] = RGammaDisc(Parameters[i],1,randState);
            sum += GammaSample[i];
        }
        for(i = 0; i < MAXPOPS; i++){
            TestQ[i+offset] = GammaSample[i]/sum;
        }
        saveRndDiscState(randState);
        ind +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
    }
}


__kernel void FillArrayWRandom(
        __global float *randomArr,
        __global uint *randGens,
        const int length
        )
{
    int pos = get_global_id(0);
    /* uint rng = getRandGen(randGens,pos); */
    uint i;
    float val;
    ulong samplesPerstream = length/get_global_size(0);
    int offset = pos*samplesPerstream;
    RndDiscState randState[1];
    initRndDiscState(randState,randGens,pos);
    if (offset < length){
        for(i = 0; i < samplesPerstream && i+offset < length; i++){
            randomArr[offset+i] = uintToUnit(rndUInt(randState));
        }
        saveRandGen(randGens,pos,randState);
    }
}

__kernel void PopNormals(
        __global float *Prev,
        __global float *norms,
        __global uint *randGens,
        const float SD,
        const int length)
{
    /* printf("Kernel: popnorms\n"); */
    RndDiscState randState[1];
    initRndDiscState(randState,randGens,0);
    int i;
    for( i=0; i < length; i+= 2){
        float2 rnorms = BoxMuller(randState);
        float oldf;
        oldf = Prev[i];
        norms[i] = rnorms.x*SD + oldf;

        if (i +1 < length){
            oldf = Prev[i+1];
            norms[i+1] = rnorms.y*SD + oldf;
        }
    }
    saveRndDiscState(randState);
}

__kernel void UpdateZ (
    __global float* Q, /* input */
    __global float* P,  /* input */
    __global int* Geno,/* input */
    __global uint* randGens, /*random numbers*/
    __global int* Z, /* output */
    __global int* error
)
{
    /* printf("Kernel: Updatez\n"); */
    int allele;
    int pop;
    int line;
    float Cutoffs[MAXPOPS];
    float sum;

    int ind = get_global_id(0);
    RndDiscState randState[1];


    while(ind < NUMINDS){
        int loc = get_global_id(1);
        while (loc < NUMLOCI) {
            initRndDiscState(randState,randGens,ind*NUMLOCI+loc);
            for (line = 0; line < LINES; line++) {
                allele = Geno[GenPos (ind,line,loc)];
                if (allele == MISSING) {   /*Missing Data */
                    Z[ZPos (ind, line, loc)] = UNASSIGNED;
                } else {
                    /*Data present */
                    sum = 0.0f;    /*compute prob of each allele being from each pop */
                    for (pop = 0; pop < MAXPOPS; pop++) {
                        Cutoffs[pop] = Q[QPos (ind, pop)] * P[PPos (loc, pop, allele)];
                        sum += Cutoffs[pop];
                    }
                    Z[ZPos (ind, line, loc)] = PickAnOptionDiscrete (MAXPOPS, sum, Cutoffs,
                                               randState);
                }
            }
            saveRndDiscState(randState);
            loc +=  get_global_size(1);// > 1 ? get_global_size(1) : 1;
        }
        ind +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
    }
}


__kernel void GetNumFromPops (
    __global int* Z, /* input */
    __global int* Geno,/* input */
    __global int* NumAlleles,/* input */
    __global int* popflags,/* input */
    __global int* NumAFromPops,/* output */
    __global int* error
)
{
    /* printf("Kernel: getnumfrompops\n"); */
    int loc = get_global_id(1);
    while (loc < NUMLOCI){
        int offset = loc*MAXPOPS*MAXALLELES;
        int ind = get_global_id(0);
        int numalleles = NumAlleles[loc];
        int pos,line,popvalue,allelevalue;
        /* initialize the NumAFromPops array */
        /* if(ind == 0){ */
        /*     for(pos = 0; pos < MAXPOPS*MAXALLELES; pos++){ */
        /*         NumAFromPops[pos+offset] = 0; */
        /*     } */
        /* } */
        /* barrier(CLK_GLOBAL_MEM_FENCE); */
        while(ind < NUMINDS) {
            if (!PFROMPOPFLAGONLY
                    || popflags[ind] == 1) {    /*individual must have popflag turned on*/
                for (line = 0; line < LINES; line++) {
                    popvalue = Z[ZPos (ind, line, loc)];
                    allelevalue = Geno[GenPos (ind, line, loc)];

                    if ((allelevalue != MISSING) && (popvalue != UNASSIGNED)) {
                        pos = NumAFromPopPos (popvalue, allelevalue) + offset;
                        AtomicInc(&NumAFromPops[pos]);
                    }

                }
            }
            ind +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
        }
        loc +=  get_global_size(1);// > 1 ? get_global_size(1) : 1;
    }
}

__kernel void UpdateP (
    __global float* P
    , __global int* NumAlleles
    , __global int* NumAFromPops
    , __global uint* randGens
    , __global int* error
    #if FREQSCORR
    , __global float *Epsilon
    , __global float *Fst
    #else
    , __global float* lambda
    #endif
)
{
    /* printf("Kernel: UpdateP\n"); */
    int loc = get_global_id(0);
    float Parameters[MAXALLELES];
    RndDiscState randState[1];
    float param;
    int allele;

    while (loc < NUMLOCI){
        int numalleles = NumAlleles[loc];
        int offset = loc*MAXPOPS*MAXALLELES;
        int pop = get_global_id(1);
        while (pop < MAXPOPS) {
            /* int pos,line,popvalue,allelevalue; */
            initRndDiscState(randState,randGens,loc*MAXPOPS +pop);

            for (allele = 0; allele < numalleles; allele++) {
                Parameters[allele] = NumAFromPops[NumAFromPopPos (pop,
                                     allele)+offset];
                #if FREQSCORR
                    param = Epsilon[EpsPos (loc, allele)]
                                     *(1.0f- Fst[pop])/Fst[pop];
                #else
                    param = lambda[pop];
                #endif
                /* if (param >= 10e20){ */
                /*     param = 10e20; */
                /* } */
                Parameters[allele] += param;
            }

            /*return a value of P simulated from the posterior Di(Parameters) */
            /*O(NumAlleles[loc]) */
            LogRDirichletDisc (Parameters, numalleles,
                               P + PPos (loc, pop, 0),
                               randState);
            saveRndDiscState(randState);
            pop +=  get_global_size(1);// > 1 ? get_global_size(1) : 1;
        }
        loc +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
    }
}



/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)
*/
__kernel void reduce(__global float *g_idata, __global float *g_odata, unsigned int n, __local volatile float *sdata)
{
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = get_local_id(0);
    unsigned int i = get_group_id(0)*(get_local_size(0)*2) + get_local_id(0);
    unsigned int gridSize = blockSize*2*get_num_groups(0);
    sdata[tid] = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        sdata[tid] += g_idata[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (i + blockSize < n){
            sdata[tid] += g_idata[i+blockSize];
        }

        i += gridSize;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction in shared mem
    if (blockSize >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } barrier(CLK_LOCAL_MEM_FENCE); }

    if (tid < 32)
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; }
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[get_group_id(0)] = sdata[0];
}



float mapLogDiffsFunc(__global float *Q, __global float *TestQ,
                       __global float *P, __global int *Geno,
                       int ind, int loc)
{
    int allele, line, pop;
    float termP;
    float termM;
    if (ind < NUMINDS && loc < NUMLOCI){
        float runningtotalP = 1.0f, runningtotalM = 1.0f;
        float logtermP = 0.0f, logtermM = 0.0f;
        for (line = 0; line < LINES; line++) {
            allele = Geno[GenPos (ind, line, loc)];
            if (allele != MISSING) {
                termP = 0.0f;
                termM = 0.0f;
                for (pop = 0; pop < MAXPOPS; pop++) {
                    termP += TestQ[QPos(ind,pop)] * P[PPos (loc, pop, allele)];
                    termM += Q[QPos(ind,pop)] * P[PPos (loc, pop, allele)];
                }

                //TODO: Evaluate underflow safe vs nonsafe
                // safe version, should not underflow
                /*if (termP > SQUNDERFLO) {
                    runningtotalP *= termP;
                } else {
                    runningtotalP *= SQUNDERFLO;
                }
                if (runningtotalP < SQUNDERFLO){
                    logtermP += log(runningtotalP);
                    runningtotalP = 1.0f;
                }

                if (termM > SQUNDERFLO) {
                    runningtotalM *= termM;
                } else {
                    runningtotalM *= SQUNDERFLO;
                }
                if (runningtotalM < SQUNDERFLO){
                    logtermM += log(runningtotalM);
                    runningtotalM = 1.0f;
                }*/
                //Might underflow?
                logtermP += log(termP);
                logtermM += log(termM);
            }
        }
        logtermP += log(runningtotalP);
        logtermM += log(runningtotalM);

        return logtermP - logtermM;
    }
    return 0.0f;
}



__kernel void mapReduceLogDiffs(__global float *Q,
                                __global float *TestQ,
                                __global float *P,
                                __global int *Geno,
                                __global float *logdiffs,
                                __global float *results,
                                __local  float *scratch)
{
    /* printf("Kernel: mapredlogdiffs\n"); */
    int ind = get_global_id(1);
    while (ind < NUMINDS){
        int numgroups = get_num_groups(0);
        int loc = get_global_id(0);
        /* idempotent */
        float logdiff = 0.0f;
        float c,y,t;
        /* Map and partial reduce */
        c = 0.0f;
        while( loc < NUMLOCI){
            float elem = mapLogDiffsFunc(Q,TestQ,P,Geno,ind,loc);
            y = elem -c;
            t = logdiff+y;
            c = (t - logdiff) - y;
            logdiff = t;
            loc +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
        }

        /* reduce locally */
        int localLoc = get_local_id(0);
        scratch[localLoc] = logdiff;
        barrier(CLK_LOCAL_MEM_FENCE);
        int devs = get_local_size(0);
        c = 0.0f;
        for(int offset = get_local_size(0) /2; offset > 0; offset >>= 1){

            if(localLoc < offset){
                y = scratch[localLoc + offset] - c;
                t = scratch[localLoc] + y;
                c = (t-scratch[localLoc]) - y;
                scratch[localLoc] = t;
            }
            //Handle if were not working on a multiple of 2
            if (localLoc == 0 && (devs-1)/2 == offset){
                y = scratch[devs-1] - c;
                t = scratch[localLoc] + y;
                c = (t-scratch[localLoc]) - y;
                scratch[localLoc] = t;
            }

            devs >>= 1;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        /* save result */
        int gid = get_group_id(0);
        if(localLoc == 0){
            results[ind*numgroups +gid] = scratch[0];
        }

        /* reduce over the groups into final result */
        barrier(CLK_GLOBAL_MEM_FENCE);
        if( localLoc==0 && gid==0 ){
            logdiffs[ind] = 0;
            for(int id =0; id < numgroups; id ++){
                logdiffs[ind] += results[ind*numgroups + id];
                results[ind*numgroups + id] = 0;
            }
        }
        ind +=  get_global_size(1);// > 1 ? get_global_size(1) : 1;
    }
}

/* copy of logdiffs, but calc only one item */
float mapLogLikeFunc(__global float *Q, __global float *P,
                       __global int *Geno, int ind, int loc)
{
    int allele, line, pop;
    float term;
    if (ind < NUMINDS && loc < NUMLOCI){
        float runningtotal = 1.0f;
        float logterm = 0.0f;
        float c = 0.0f;
        float y,t;
        for (line = 0; line < LINES; line++) {
            allele = Geno[GenPos (ind, line, loc)];
            if (allele != MISSING) {
                term = 0.0f;
                float d = 0.0f;
                float z,u;
                for (pop = 0; pop < MAXPOPS; pop++) {

                    z = Q[QPos(ind,pop)] * P[PPos (loc, pop, allele)] - d;
                    u = term + z;
                    d = (u - term) - z;
                    term = u;
                }

                //TODO: Evaluate underflow safe vs nonsafe
                // safe version, should not underflow
                /*
                if (term > SQUNDERFLO) {
                    runningtotal *= term;
                } else {
                    runningtotal *= SQUNDERFLO;
                }
                if (runningtotal < SQUNDERFLO){
                    logterm += log(runningtotal);
                    runningtotal = 1.0f;
                }*/
                //Might underflow?
                y = log(term) - c;
                t = logterm + y;
                c = (t - logterm) -y;
                logterm = t;
            }
        }
        logterm += log(runningtotal);
        return logterm;
    }
    return 0.0f;
}

__kernel void mapReduceLogLike(__global float *Q,
                                __global float *P,
                                __global int *Geno,
                                __global float *loglikes,
                                __global float *results,
                                __local  float *scratch)
{

    /* printf("Kernel: mapredloglike %d\n",0); */
    int ind = get_global_id(1);
    while (ind < NUMINDS){
        int loc = get_global_id(0);
        int numgroups = get_num_groups(0);
        /* idempotent */
        float logterm = 0.0f;
        float c,y,t;
        /* Map and partial reduce */
        /* clear results buffer */

        if (ind < NUMINDS && loc < NUMLOCI){
            c = 0.0f;
            while( loc < NUMLOCI){
                float elem = mapLogLikeFunc(Q,P,Geno,ind,loc);
                y = elem - c;
                t = logterm + y;
                c = (t - logterm) - y;
                logterm = t;
                loc +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
            }

            /* reduce locally */
            int localLoc = get_local_id(0);
            scratch[localLoc] = logterm;
            barrier(CLK_LOCAL_MEM_FENCE);
            int devs = get_local_size(0);
            c = 0.0f;
            for(int offset = get_local_size(0) /2; offset > 0; offset >>= 1){
                if(localLoc < offset){
                    y = scratch[localLoc + offset] - c;
                    t = scratch[localLoc] + y;
                    c = (t-scratch[localLoc]) - y;
                    scratch[localLoc] = t;
                }
                //Handle if were not working on a multiple of 2
                if (localLoc == 0 && (devs-1)/2 == offset){
                    y = scratch[devs-1] - c;
                    t = scratch[localLoc] + y;
                    c = (t-scratch[localLoc]) - y;
                    scratch[localLoc] = t;
                }

                devs >>= 1;
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            /* save result */
            int gid = get_group_id(0);
            if(localLoc == 0){
                /*results[ind*numgroups +gid] = 1;*/
                results[ind*numgroups +gid] = scratch[0];
            }

            /* reduce over the groups into final result */
            barrier(CLK_GLOBAL_MEM_FENCE);
            if(localLoc==0 && gid==0 ){
                loglikes[ind] = 0;
                for(int id =0; id < numgroups; id ++){
                    loglikes[ind] += results[ind*numgroups + id];
                    results[ind*numgroups + id] = 0;
                }
            }
        }
        ind +=  get_global_size(1);// > 1 ? get_global_size(1) : 1;
    }
}

__kernel void CalcLike(
        __global float *loglikes,
        __global float *indlike_norm,
        __global float *sumindlike,
        const int usesumindlike, /* this is true after burnin */
        __global float *loglike,
        __global float *results,
        __local  float *scratch
        )
{
    /* printf("Kernel: calclike\n"); */
    int ind = get_global_id(0);

    float logterm = 0.0f;
    int numgroups = get_num_groups(0);
    float c,y,t;

    /* Map and partial reduce */
    c = 0.0f;
    while( ind < NUMINDS){
        float elem = loglikes[ind];
        if (usesumindlike) {
            if (indlike_norm[ind]==0.0f) {
                indlike_norm[ind] = elem;
            }
            sumindlike[ind] += exp(elem-indlike_norm[ind]);
        }
        y = elem - c;
        t = logterm + y;
        c = (t - logterm) - y;
        logterm = t;
        ind +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
    }

    int localLoc = get_local_id(0);
    scratch[localLoc] = logterm;
    barrier(CLK_LOCAL_MEM_FENCE);
    int devs = get_local_size(0);
    c = 0.0f;
    for(int offset = get_local_size(0) /2; offset > 0; offset >>= 1){
        if(localLoc < offset){
            y = scratch[localLoc + offset] - c;
            t = scratch[localLoc] + y;
            c = (t-scratch[localLoc]) - y;
            scratch[localLoc] = t;
        }
        //Handle if were not working on a multiple of 2
        if (localLoc == 0 && (devs-1)/2 == offset){
            y = scratch[devs-1] - c;
            t = scratch[localLoc] + y;
            c = (t-scratch[localLoc]) - y;
            scratch[localLoc] = t;
        }

        devs >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* save result */
    int gid = get_group_id(0);
    if(localLoc == 0){
        results[gid] = scratch[0];
    }

    /* reduce over the groups into final result */
    barrier(CLK_GLOBAL_MEM_FENCE);
    if(localLoc==0 && gid==0 ){
        loglike[0] = 0;
        for(int id =0; id < numgroups; id++){
            loglike[0] += results[id];
            results[id] = 0;
        }
    }
}

__kernel void MetroAcceptTest(
        __global float *TestQ,
        __global float *Q,
        __global uint *randGens,
        __global float *logdiffs,
        __global int *popflags)
{
    /* printf("Kernel: metacctest\n"); */
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
        ind +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
    }
}

__kernel void GetNumLociPops(
        __global int *Z,
        __global int *popflags,
        __global int *NumLociPops)
{
    /* printf("Kernel: getnumlocipops\n"); */
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
            loc +=  get_global_size(1);// > 1 ? get_global_size(1) : 1;
        }
        ind +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
    }
}

__kernel void UpdQDirichlet(
        __global float *Alpha,
        __global int *NumLociPops,
        __global uint *randGens,
        __global float *Q,
        __global int *popflags)
{
    /* printf("Kernel: updqdir\n"); */
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
            ind +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
        }
    }
}


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
    /* printf("Kernel: updatefst\n"); */
    int pop = get_global_id(1);
    int numgroups = get_num_groups(0);
    float c, y, t;// KahanSum
    while (pop < numfst){
        int loc = get_global_id(0);
        float newf = normals[pop];
        /* ensure newf is large enough so we don't cause over/underflow */
        if (newf > DELTA && newf < 1.0f - DELTA){
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
                loc +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
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
            if(localLoc==0&&gid==0){
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
        pop +=  get_global_size(1);// > 1 ? get_global_size(1) : 1;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}


float AlphaPriorDiff (float newalpha, float oldalpha)
{
    /*returns log diff in priors for the alpha, assuming a gamma prior on alpha
      See notes 7/29/99 */
    return ((ALPHAPRIORA - 1) * log (newalpha / oldalpha) +
            (oldalpha - newalpha) / ALPHAPRIORB);
}


__kernel void UpdateAlpha(
       __global float *Q,
       __global float *Alpha,
       __global int *popflags,
       __global float *norms,
       __global float *results,
       __global uint *randGens,
       __local float *scratch,
       const int POPFLAGINDS)
{
    /* printf("Kernel: Update alpha\n"); */
    int alpha = get_global_id(1);
    float c, y, t; // KahanSum
    while( alpha < NUMALPHAS){
        int ind = get_global_id(0);
        if(ind < NUMINDS){
            int redpop;
            int numredpops = MAXPOPS;

            float newalpha = norms[alpha];
            float oldalpha = Alpha[alpha];
            float alphasum =0.0f;

            if ((newalpha > 0) && ((newalpha < ALPHAMAX) || (!(UNIFPRIORALPHA)) ) ) {
                if (POPALPHAS){ numredpops = alpha +1; }
                //TODO: Evaluate underflow safe vs nonsafe
                float sum = 1.0f;
                float total = 0.0f;
                c = 0.0f;
                while( ind < NUMINDS){
                    if (!((USEPOPINFO) && (popflags[ind]))) {
                        //Safe version (similar to in code)
                        //Watching out for underflow
                        float elem = 1.0f;
                        for(redpop = alpha; redpop < numredpops; redpop++){
                            elem *= Q[QPos (ind, redpop)];
                        }

                        if (elem > SQUNDERFLO){
                            sum *= elem;
                        } else {
                            sum *= SQUNDERFLO;
                        }
                        if(sum < SQUNDERFLO){
                            y = log(sum)- c;
                            t = total + y;
                            c = (t-total) - y;
                            total = t;
                            sum = 1.0f;
                        }
                        //Might underflow?
                        /* float elem = 0.0f; */
                        /* for(redpop = alpha; redpop < numredpops; redpop++){ */
                        /*     elem += log(Q[QPos (ind, redpop)]); */
                        /* } */
                        /* y = elem - c; */
                        /* t = total + y; */
                        /* c = (t-total) - y; */
                        /* total = t; */

                        ind +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
                    }
                }

                total += log(sum);

                /* reduce locally */
                int localId = get_local_id(0);
                scratch[localId] = total;
                barrier(CLK_LOCAL_MEM_FENCE);
                int devs = get_local_size(0);
                c = 0.0f;
                for(int offset = get_local_size(0) /2; offset > 0; offset >>= 1){
                    if(localId < offset){
                        y = scratch[localId + offset] - c;
                        t = scratch[localId] + y;
                        c = (t-scratch[localId]) - y;
                        scratch[localId] = t;
                    }
                    //Handle if were not working on a multiple of 2
                    if (localId == 0 && (devs-1)/2 == offset){
                        y = scratch[devs-1] - c;
                        t = scratch[localId] + y;
                        c = (t-scratch[localId]) - y;
                        scratch[localId] = t;
                    }

                    devs >>= 1;
                    barrier(CLK_LOCAL_MEM_FENCE);
                }

                int numgroups = get_num_groups(0);
                int gid = get_group_id(0);

                if(localId == 0){
                    results[alpha*numgroups +gid] = scratch[0];
                }
                //TODO: Handle if numgroups are more than MAXGROUPS
                //Possibly by reducing with global barrier.

                barrier(CLK_GLOBAL_MEM_FENCE);
                if(localId == 0&& gid==0 ){
                    RndDiscState randState[1];
                    initRndDiscState(randState,randGens,alpha);
                    for (int i=0; i<MAXPOPS; i++)  {
                        alphasum += Alpha[i];
                    }
                    float logprobdiff = 0.0f;
                    float logterm = 0.0f;
                    if (!(UNIFPRIORALPHA)) logprobdiff = AlphaPriorDiff (newalpha, oldalpha);

                    for(int id =0; id < numgroups; id++){
                        logterm += results[alpha*numgroups + id];
                        results[alpha*numgroups + id] = 0;
                    }

                    int multiple = numredpops - alpha;
                    float lpsum = (newalpha - oldalpha) * logterm;
                    /*lpsum -= (oldalpha - 1.0f) * logterm;
                      lpsum += (newalpha - 1.0f) * logterm;*/

                    float sumalphas = alphasum;
                    if (POPALPHAS){
                        sumalphas += newalpha - oldalpha;
                    } else {
                        sumalphas = MAXPOPS*newalpha;
                    }

                    lpsum += ((lgamma(sumalphas) - lgamma(alphasum)) - multiple*(lgamma(newalpha) - lgamma(oldalpha))) * POPFLAGINDS;
                    
                    /*lpsum -= (lgamma (alphasum) - multiple * lgamma ( oldalpha)) * POPFLAGINDS;
                      lpsum += (lgamma (sumalphas) - multiple * lgamma ( newalpha)) * POPFLAGINDS;*/
                    logprobdiff += lpsum;
                    /* printf("UpdateAlpha %d %f %f %d %d %d %d\n",alpha,logprobdiff,newalpha,get_group_id(0), get_group_id(1), get_num_groups(0), get_num_groups(1)); */
                    if (rndDisc(randState) < exp(logprobdiff)) {   /*accept new f */
                        for(redpop = alpha; redpop < numredpops; redpop++){
                            Alpha[redpop] = newalpha;
                        }
                    }
                    saveRndDiscState(randState);
                }
            }
        }
        alpha +=  get_global_size(1);// > 1 ? get_global_size(1) : 1;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}





__kernel void NonIndUpdateEpsilon(
        __global float *P,
        __global float *Epsilon,
        __global float *Fst,
        __global int *NumAlleles,
        __global uint *randGens,
        __global float *lambdas,
        const float invsqrtnuminds)
{
    /* printf("Kernel: nonindupdeps\n"); */

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
                if (fabs(lambda - 1.0f) > DELTA) {              /* compute prior ratio */
                    /* as it is in code */
                    /* float ratio = (eps1 + diff)* (eps2 - diff)/(eps1)/(eps2) */
                    /* sum += log(pow(ratio, lambda-1.0f)); */
                    /* as it probably should be ? */
                    float ratio = (eps1 + diff)* (eps2 - diff)/(eps1*eps2);
                    sum += (lambda-1.0f)*log(ratio);
                }
                float randVal = rndDisc(randState);
                if (randVal < exp(sum) || log(randVal) < sum ){
                    AtomicAdd(&Epsilon[EpsPos(loc,allele1)],diff);
                    AtomicAdd(&Epsilon[EpsPos(loc,allele2)],-diff);
                }
            }
        }
        loc +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    saveRndDiscState(randState);
}


void UpdateSumsPop (
        __global float *lambda,
        __global float *sumlambda,
        __global float *Fst,
        __global float *FstSum,
        int pop)
{
    sumlambda[pop] += lambda[pop];
    if (FREQSCORR) {
        FstSum[pop] += Fst[pop];
    }
}

/* Data collection paralell over pop */
__kernel void DataCollectPop(
       __global float *Alpha,
       __global float *AlphaSum,
       __global float *lambda,
       __global float *lambdaSum,
       __global float *Fst,
       __global float *FstSum
       )
{
    /* printf("Kernel: datacp\n"); */
    int pop = get_global_id(0);
    while (pop < MAXPOPS) {
        UpdateSumsPop(lambda,lambdaSum,Fst,FstSum,pop);
        int pos,loc;
        if (LOCPRIOR && NOADMIX==0) {
                for (loc=0; loc<=NUMLOCATIONS; loc++) {
                    pos = AlphaPos(loc, pop);
                    AlphaSum[pos] += Alpha[pos];
                }
        } else if (!(NOADMIX) && (!(NOALPHA))) {
                AlphaSum[pop] += Alpha[pop];
        }

        /*if (pop == 0){
            if (LOCPRIOR)
                for (i=0; i<LocPriorLen; i++) {
                    sumLocPrior[i] += LocPrior[i];
                }
        }*/
        pop +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
    }
}

/* Data collection paralell over ind */
void UpdateSumsInd (
        __global float *Q,
        __global float *QSum,
        __global int *AncestDist,
        int pop,
        int ind
        )
{

    QSum[QPos (ind, pop)] += Q[QPos (ind, pop)];
    if (ANCESTDIST) {
        int box = ((int) (Q[QPos (ind, pop)] * ((float) NUMBOXES)));
        /*printf("Kernel: %1.3f__%d  ",Q[QPos(ind,pop)],box); */
        if (box == NUMBOXES) {
            box = NUMBOXES - 1;    /*ie, Q = 1.000 */
        }
        AncestDist[AncestDistPos (ind, pop, box)]++;
    }
}

__kernel void DataCollectInd(
        __global float *Q,
        __global float *QSum,
        __global int *AncestDist
        )
{
    /* printf("Kernel: dataci\n"); */
    int pop = get_global_id(0);
    while(pop < MAXPOPS){
        int ind = get_global_id(1);
        while (ind < NUMINDS){
            UpdateSumsInd(Q,QSum,AncestDist,pop,ind);
            ind +=  get_global_size(1);// > 1 ? get_global_size(1) : 1;
        }
        pop +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
    }
}

/* Data collection parallell over loc */
void UpdateSumsLoc (
        __global int *NumAlleles,
        __global float *P,
        __global float *PSum,
        __global float *Epsilon,
        __global float *SumEpsilon,
        int pop,
        int loc
        )
{
    int allele;
    for (allele = 0; allele < NumAlleles[loc]; allele++) {
        PSum[PPos (loc, pop, allele)] += P[PPos (loc, pop, allele)];
    }
    if (FREQSCORR){
        for (allele = 0; allele < NumAlleles[loc]; allele++) {
            SumEpsilon[EpsPos (loc, allele)] += Epsilon[EpsPos (loc, allele)];
        }
    }
}

__kernel void DataCollectLoc(
        __global int *NumAlleles,
        __global float *P,
        __global float *PSum,
        __global float *Epsilon,
        __global float *SumEpsilon
        ) 
{
    /* printf("Kernel: datacl\n"); */
    int pop = get_global_id(0);
    while (pop < MAXPOPS) {
        int loc = get_global_id(1);
        while (loc < NUMLOCI){
            UpdateSumsLoc(NumAlleles,P,PSum,Epsilon,SumEpsilon,pop,loc);
            loc +=  get_global_size(1);// > 1 ? get_global_size(1) : 1;
        }
        pop +=  get_global_size(0);// > 1 ? get_global_size(0) : 1;
    }
}

__kernel void ComputeProbFinish(
        __global float *loglike,
        __global float *sumlikes,
        __global float *sumsqlikes
        )
{

    /* printf("Kernel: computeprobfin\n"); */
    float like = loglike[0];
    sumlikes[0] += like;
    sumsqlikes[0] += like*like;
}
