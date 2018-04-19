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
