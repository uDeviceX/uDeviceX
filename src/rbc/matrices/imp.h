struct Matrices;
struct Coords;

// tag::interface[]
void matrices_read(const char *path, /**/ Matrices**);
void matrices_read_filter(const char *path, const Coords*, /**/ Matrices**);

void matrices_get(const Matrices*, int i, /**/ double**);
void matrices_get_r(const Matrices*, int i, /**/ double r[3]);
int  matrices_get_n(const Matrices*);
void matrices_log(const Matrices*);

void matrices_fin(Matrices*);
// end::interface[]
