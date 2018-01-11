struct RbcStretch;

void rbc_stretch_ini(const char* path, int nv, /**/ RbcStretch **fp); /* `nv` is for error check */
void rbc_stretch_fin(RbcStretch *f);
void rbc_stretch_apply(int nm, const RbcStretch*, /**/ Force*);
