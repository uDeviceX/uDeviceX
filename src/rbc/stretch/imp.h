namespace rbc { namespace stretch {

/* force */
struct StretchForce;

void ini(const char* path, int nv, /**/ StretchForce **fp); /* `nv` is for error check */
void fin(StretchForce *f);
void apply(int nm, const StretchForce*, /**/ Force*);

}} /* namespace */
