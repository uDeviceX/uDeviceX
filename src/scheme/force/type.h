/* "view" structures: to be passed to device funtions */

/* constant force f */
struct BForce_cste_v {
    float3 a;
};

/* double poiseuille */
struct BForce_dp_v {
    float a;
};

/* shear force fx = a * (y - yc) */
struct BForce_shear_v {
    float a;
};

/* 4 rollers mill */
struct BForce_rol_v {
    float a;
};

/* radial force decaying as 1/r */
struct BForce_rad_v {
    float a;
};

enum {
    BODY_FORCE_V_NONE,
    BODY_FORCE_V_CSTE,
    BODY_FORCE_V_DP,
    BODY_FORCE_V_SHEAR,
    BODY_FORCE_V_ROL,
    BODY_FORCE_V_RAD
};

union BForceParam_v {
    BForce_cste_v cste;
    BForce_dp_v dp;
    BForce_shear_v shear;
    BForce_rol_v rol;
    BForce_rad_v rad;
};

struct BForce_v {
    BForceParam_v p;
    int type;
};

/* structure containing parameters on host */

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
