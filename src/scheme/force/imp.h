struct Particle;
struct Force;
struct BForce;
struct float3;

// tag::mem[]
void bforce_ini(BForce **p);
void bforce_fin(BForce *p);
// end::mem[]

// tag::ini[]
void bforce_ini_none(/**/ BForce *p);
void bforce_ini_cste(float3 f, /**/ BForce *p);
void bforce_ini_dp(float a, /**/ BForce *p);
void bforce_ini_shear(float a, /**/ BForce *p);
void bforce_ini_rol(float a, /**/ BForce *p);
void bforce_ini_rad(float a, /**/ BForce *p);
// end::ini[]

// tag::interface[]
void bforce_adjust(float3 f, /**/ BForce *fpar);
void bforce_apply(long it, Coords c, float mass, const BForce *bf, int n, const Particle *pp, /**/ Force *ff);
// end::interface[]
