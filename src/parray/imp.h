struct PaArray;

struct PaArray_v;
struct PaCArray_v;

struct Particle;

void parray_ini(PaArray **);
void parray_fin(PaArray *);

void parray_push(const Particle *pp, PaArray *a);
void parray_push_color(const Particle *pp, const int *cc, PaArray *a);

bool parray_is_colored(const PaArray *a);

void parray_get_view(const PaArray *a, PaArray_v *v);
void parray_get_view(const PaArray *a, PaCArray_v *v);

