struct Particle;
struct Force;
struct BForce;

// tag::params[]
/* constant force f */
struct BForce_cste {
    float3 a; // acceleration vector
};

/* double poiseuille */
struct BForce_dp {
    float a; // acceleration in x direction
};

/* shear force fx = a * (y - yc) */
struct BForce_shear {
    float a; // acceleration is a * y in x direction
};

/* 4 rollers mill */
struct BForce_rol {
    float a;  // intensity of the acceleration
};

/* radial force decaying as 1/r */
struct BForce_rad {
    float a; // radial acceleartion is a / r
};
// end::params[]

// tag::mem[]
void bforce_ini(BForce **p);
void bforce_fin(BForce *p);
// end::mem[]

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
void bforce_apply(long it, Coords c, float mass, const BForce *bf, int n, const Particle *pp, /**/ Force *ff);
// end::interface[]
