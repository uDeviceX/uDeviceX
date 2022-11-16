/* structure containing parameters on host */

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

enum {
    BODY_FORCE_NONE,
    BODY_FORCE_CSTE,
    BODY_FORCE_DP,
    BODY_FORCE_SHEAR,
    BODY_FORCE_ROL,
    BODY_FORCE_RAD
};

union BForceParam {
    BForce_cste cste;
    BForce_dp dp;
    BForce_shear shear;
    BForce_rol rol;
    BForce_rad rad;
};

struct BForce {
    BForceParam p;
    int type;
};
