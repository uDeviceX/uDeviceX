struct RbcStretch;
struct Force;

void rbc_stretch_ini(const char* path, int nv, /**/ RbcStretch**); /* `nv` is for error check */
void rbc_stretch_fin(RbcStretch*);
void rbc_stretch_apply(int nm, const RbcStretch*, /**/ Force*);
