namespace rbc { namespace stretch {

/* force */
struct Fo;

void ini(const char* path, int nv, /**/ Fo **fp); /* `nv` is for error check */
void fin(Fo *f);
void apply(int nm, const Particle*, const Fo*, /**/ Force*);

}} /* namespace */
