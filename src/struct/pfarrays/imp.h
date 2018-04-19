struct PFarrays;

struct Parray;
struct Farray;

void pfarrays_ini(PFarrays**);
void pfarrays_fin(PFarrays*);

void pfarray_clear(PFarrays*);
void pfarray_push(PFarrays*, long n, Parray, Farray);
int pfarray_size(const PFarrays*);
void pfarray_get(const PFarrays*, long *n, Parray*, Farray*);
