void ini_none(/**/ FParam *p);
void ini(FParam_cste_v par, /**/ FParam *p);
void ini(FParam_dp_v par, /**/ FParam *p);
void ini(FParam_shear_v par, /**/ FParam *p);
void ini(FParam_rol_v par, /**/ FParam *p);
void ini(FParam_rad_v par, /**/ FParam *p);

void body_force(Coords c, float mass, FParam fpar, int n, const Particle *pp, /**/ Force *ff);

/* not polymorphic for now        */
/* related to velocity controller */
void adjust(float3 f, /**/ FParam *fpar);
