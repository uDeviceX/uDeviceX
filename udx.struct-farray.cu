// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/struct/farray/imp.cpp */
#include <stdio.h>

/* begin src/struct/farray/imp.h */
struct FoArray {
    bool stress;
    float *ff;
    float *ss; /* stress */
};

struct FoArray_v;
struct FoSArray_v;

struct Force;

// tag::push[]
void farray_push_ff(Force *ff, FoArray *a);
void farray_push_ss(float *ss, FoArray *a);
// end::push[]

void farray_clear(FoArray *);

// tag::get[]
bool farray_has_stress(const FoArray *a);
// end::get[]

// tag::view[]
void farray_get_view(const FoArray *a, FoArray_v *v);  // <1>
void farray_get_view(const FoArray *a, FoSArray_v *v); // <2>
// end::view[]
/* end src/struct/farray/imp.h */

/* begin src/struct/farray/type.h */
struct FoArray_v {
    float *ff;
};

struct FoSArray_v {
    float *ff;
    float *ss;
};
/* end src/struct/farray/type.h */
/* begin src/struct/farray/imp/main.h */
void farray_push_ff(Force *ff, FoArray *a) {
    a->ff = (float*) ff;
    a->stress = false;
}

void farray_push_ss(float *ss, FoArray *a) {
    a->ss = ss;
    a->stress = true;
}

bool farray_has_stress(const FoArray *a) {
    return a->stress;
}

void farray_get_view(const FoArray *a, FoArray_v *v) {
    v->ff = a->ff;
}

void farray_get_view(const FoArray *a, FoSArray_v *v) {
    v->ff = a->ff;
    v->ss = a->ss;
}
/* end src/struct/farray/imp/main.h */

/* end src/struct/farray/imp.cpp */

#line 1 "src/struct/parray/imp.cpp"
