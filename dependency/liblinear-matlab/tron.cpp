#include <mex.h>
#include <extern/include/blas.h>

#ifdef __cplusplus
extern "C" {
#endif

double dnrm2_wrapper(ptrdiff_t *a, double *b, ptrdiff_t *c) {
    return dnrm2_(a,b,c);
}
double ddot_wrapper(ptrdiff_t *a, double *b, ptrdiff_t *c, double *d, ptrdiff_t *e) {
    return ddot_(a,b,c,d,e);
}

void daxpy_wrapper(ptrdiff_t *a, double *b, double *c, ptrdiff_t *d, double *e, ptrdiff_t *f) {
    return daxpy_(a,b,c,d,e,f);
}

void dscal_wrapper(ptrdiff_t *a, double *b, double *c, ptrdiff_t *d) {
    return dscal(a,b,c,d);
}


#define dnrm2_ __dnrm2_
#define ddot_  __ddot_
#define daxpy_ __daxpy_
#define dscal_ __dscal_

typedef ptrdiff_t INT_T;

double dnrm2_(int *, double *, int *);
double ddot_(int *, double *, int *, double *, int *);
int daxpy_(int *, double *, double *, int *, double *, int *);
int dscal_(int *, double *, double *, int *);

double dnrm2_(int *a, double *b, int *c) {
    INT_T a1 = *a; INT_T c1 = *c;
    return dnrm2_wrapper( &a1,b,&c1 );
}
double ddot_(int *a, double *b, int *c, double *d, int *e) {
    INT_T a1 = *a; INT_T c1 = *c; INT_T e1 = *e;
    return ddot_wrapper( &a1,b,&c1,d,&e1 );
}
int daxpy_(int *a, double *b, double *c, int *d, double *e, int *f) {
    INT_T a1 = *a; INT_T d1 = *d; INT_T f1 = *f;
    daxpy_wrapper( &a1,b,c,&d1,e,&f1 );
    return 0;
}
int dscal_(int *a, double *b, double *c, int *d){
    INT_T a1 = *a; INT_T d1 = *d;
    dscal_wrapper( &a1,b,c,&d1 );
    return 0;
}

#ifdef __cplusplus
}
#endif

#include "../liblinear-default/tron.cpp"

