struct Particle;
struct Force;
struct BForce;
struct float3;

// tag::mem[]
void bforce_ini(BForce **p);
void bforce_fin(BForce *p);
// end::mem[]

// tag::ini[]
void bforce_ini_none(/**/ BForce *p);            // <1>
void bforce_ini_cste(float3 f, /**/ BForce *p);  // <2>
void bforce_ini_dp(float a, /**/ BForce *p);     // <3>
void bforce_ini_shear(float a, /**/ BForce *p);  // <4>
void bforce_ini_rol(float a, /**/ BForce *p);    // <5>
void bforce_ini_rad(float a, /**/ BForce *p);    // <6>
// end::ini[]

// tag::interface[]
void bforce_adjust(float3 f, /**/ BForce *fpar); // <1>
void bforce_apply(long it, Coords c, float mass, const BForce *bf, int n, const Particle *pp, /**/ Force *ff); // <2>
// end::interface[]
