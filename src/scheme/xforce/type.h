/* structures to be passed to the device */

/* constant force f */
struct FParam_cste_d {
    float3 a;
};

/* double poiseuille */
struct FParam_dp_d {
    float a;
};

/* shear force fx = a * (y - yc) */
struct FParam_shear_d {
    float a;
};

/* 4 rollers mill */
struct FParam_rol_d {
    float a;
};

enum {
    TYPE_NONE,
    TYPE_CSTE,
    TYPE_DP,
    TYPE_SHEAR,
    TYPE_ROL
};

union FParam_d {
    FParam_cste_d cste;
    FParam_dp_d dp;
    FParam_shear_d shear;
    FParam_rol_d rol;        
};

/* structure containing parameters on host */

struct FParam {
    FParam_d dev;
    int type;
};
