// tag::ini[]
void bforce_ini_none(/**/ BForce *p);
void bforce_ini(BForce_cste par, /**/ BForce *p);
void bforce_ini(BForce_dp par, /**/ BForce *p);
void bforce_ini(BForce_shear par, /**/ BForce *p);
void bforce_ini(BForce_rol par, /**/ BForce *p);
void bforce_ini(BForce_rad par, /**/ BForce *p);
// end::ini[]

// tag::interface[]
void bforce_adjust(float3 f, /**/ BForce *fpar);

void bforce_get_view(long it, BForce bforce, /**/ BForce_v *view);
void bforce_apply(Coords c, float mass, BForce_v fpar, int n, const Particle *pp, /**/ Force *ff);
// end::interface[]
