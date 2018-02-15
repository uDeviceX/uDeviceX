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

// tag::get[]
bool farray_has_stress(const FoArray *a);
// end::get[]

// tag::view[]
void farray_get_view(const FoArray *a, FoArray_v *v);  // <1>
void farray_get_view(const FoArray *a, FoSArray_v *v); // <2>
// end::view[]
