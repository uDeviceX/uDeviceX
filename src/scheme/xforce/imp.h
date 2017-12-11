void ini_none(/**/ FParam *p);
void ini(FParam_cste_d par, /**/ FParam *p);
void ini(FParam_dp_d par, /**/ FParam *p);
void ini(FParam_shear_d par, /**/ FParam *p);
void ini(FParam_rol_d par, /**/ FParam *p);

void body_force(Coords c, float mass, FParam fpar, int n, const Particle *pp, /**/ Force *ff);

/* not polymorphic for now        */
/* related to velocity controller */
void adjust(float3 f, /**/ FParam *fpar);
