struct Matrices;
struct Coords;

// tag::interface[]
void matrices_read(const char *path, /**/ Matrices**);
void matrices_read_filter(const char *path, const Coords*, /**/ Matrices**);

void matrices_get(Matrices*, int i, /**/ double**);
void matrices_get_r(Matrices*, int i, /**/ double r[3]);
int  matrices_get_n(Matrices*);
void matrices_log(Matrices*);

void matrices_fin(Matrices*);
// end::interface[]
