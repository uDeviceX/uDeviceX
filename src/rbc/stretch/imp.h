namespace rbc { namespace stretch {

/* force */
struct StretchForce;

void rbc_stretch_ini(const char* path, int nv, /**/ StretchForce **fp); /* `nv` is for error check */
void rbc_stretch_fin(StretchForce *f);
void rbc_stretch_apply(int nm, const StretchForce*, /**/ Force*);

}} /* namespace */
