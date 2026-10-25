// generated from udx.flat.cu
#define DEV_CUDA
#define CC_RELEASE
#define KL_RELEASE
#include "udx.api.cuh"

/* begin src/struct/parray/imp.cpp */
#include <stdio.h>

/* begin src/struct/parray/imp.h */
struct PaArray {
    bool colors;
    const float *pp;
    const int *cc;
};

struct PaArray_v;
struct PaCArray_v;

struct Particle;

// tag::push[]
void parray_push_pp(const Particle *pp, PaArray *a);
void parray_push_cc(const int *cc, PaArray *a);
// end::push[]

// tag::get[]
bool parray_is_colored(const PaArray *a);
// end::get[]

// tag::view[]
void parray_get_view(const PaArray *a, PaArray_v *v);  // <1>
void parray_get_view(const PaArray *a, PaCArray_v *v); // <2>
// end::view[]
/* end src/struct/parray/imp.h */

/* begin src/struct/parray/type.h */
struct PaArray_v {
    const float *pp;
};

struct PaCArray_v {
    const float *pp;
    const int *cc;   /* colors */
};
/* end src/struct/parray/type.h */
/* begin src/struct/parray/imp/main.h */
void parray_push_pp(const Particle *pp, PaArray *a) {
    a->pp = (const float*)pp;
    a->colors = false;
};

void parray_push_cc(const int *cc, PaArray *a) {
    a->cc = cc;
    a->colors = true;
}

bool parray_is_colored(const PaArray *a) {
    return a->colors;
}

void parray_get_view(const PaArray *a, PaArray_v *v) {
    v->pp = a->pp;
}

void parray_get_view(const PaArray *a, PaCArray_v *v) {
    v->pp = a->pp;
    v->cc = a->cc;
}
/* end src/struct/parray/imp/main.h */

/* end src/struct/parray/imp.cpp */

#line 1 "src/struct/pfarrays/imp.cpp"
