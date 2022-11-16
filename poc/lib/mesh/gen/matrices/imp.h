struct Matrices;
struct Coords;

// tag::interface[]
void matrices_read(const char *path, /**/ Matrices**); // <1>
void matrices_read_filter(const char *path, const Coords*, /**/ Matrices**); // <2>

void matrices_get(const Matrices*, int i, /**/ double**); // <3>
void matrices_get_r(const Matrices*, int i, /**/ double r[3]); // <4>
int  matrices_get_n(const Matrices*); // <5>
void matrices_log(const Matrices*);   // <6>

void matrices_fin(Matrices*);
// end::interface[]
