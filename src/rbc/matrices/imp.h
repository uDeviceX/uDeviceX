struct Matrices;
struct Coords;

// tag::interface[]
void matrices_read(const char *path, /**/ Matrices**);
void matrices_read_r(const char *path, /**/ Matrices**);
void matrices_ini_filter(Matrices*, Coords*, /**/ Matrices**);

void matrices_get(Matrices*, int i, /**/ double M[4*4]);
void matrices_get_r(Matrices*, int i, /**/ double r[3]);
int  matrices_get_n(Matrices*);

void matrices_fin(Matrices*);
// end::interface[]
