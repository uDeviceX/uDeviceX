struct PFarrays;

struct PaArray;
struct FoArray;

void pfarrays_ini(PFarrays**);
void pfarrays_fin(PFarrays*);

void pfarray_clear(PFarrays*);
void pfarray_push(PFarrays*, long n, PaArray, FoArray);
int  pfarray_size(const PFarrays*);
void pfarray_get(int i, const PFarrays*, long *n, PaArray*, FoArray*);
