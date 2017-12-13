void ini_none(/**/ BForce *p);
void ini(BForce_cste par, /**/ BForce *p);
void ini(BForce_dp par, /**/ BForce *p);
void ini(BForce_shear par, /**/ BForce *p);
void ini(BForce_rol par, /**/ BForce *p);
void ini(BForce_rad par, /**/ BForce *p);

/* TODO not polymorphic for now   */
/* related to velocity controller */
void adjust(float3 f, /**/ BForce *fpar);

void get_view(long it, BForce bforce, /**/ BForce_v *view);
void body_force(Coords c, float mass, BForce_v fpar, int n, const Particle *pp, /**/ Force *ff);
