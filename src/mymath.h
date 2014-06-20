#include "randGen.h"
extern double Square(double x);
extern double SampleVar(double sumsq,double sum,long num);
extern int PickAnOption(int total,double sum,double cutoffs[]);
extern int PickAnOptionDiscrete(int total,double sum,double Probs[],
                                RndDiscState *randState);
extern double SD(double sumsq, double sum, long num);
extern double LDirichletProb(double prior[],double post[],int length);
extern double LGammaDistProb(double alpha,double beta, double y);
extern double FindAveLogs(double *logmax,double *sum, double lognext,int rep);
extern void RandomOrder(int list[],int length);
extern double Factorial(int n);
extern double mylgamma(double z);
extern double ChiSq(int *list1,int len1,int *list2,int len2,int mincount,
                    int missing,int *df);

