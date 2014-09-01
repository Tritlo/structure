#ifndef RANDGEN
#define RANDGEN
/* #define MAXRANDVAL 4294967296 */
#define MAXRANDVAL 281474976710656

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

uint getRandUint(RndDiscState *state){
    uint a = 25214903917;
    uint c = 11;
    uint m = 48;
    uint x = state->rng;
    uint xn = (a*x + c) >> 48;
    state->rng = xn;
    return xn;
}

/* uint getRandUint(RndDiscState *state){ */
/* return MWC64X_NextUint(&(state->rng)); */
/* } */

float rndDisc(RndDiscState * state)
{
    uint rand = getRandUint(state);
    float val = uintToUnit(rand);
    return val;
}

uint rndUInt(RndDiscState * state)
{
    return getRandUint(state);
}

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

