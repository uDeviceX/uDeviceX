/* structures to be passed to the device */

/* constant force f */
struct FParam_cste_d {
    float3 f;
};

/* double poiseuille */
struct FParam_dp_d {
    float f;
};

/* shear force fx = a * (y - yc) */
struct FParam_shear_d {
    float a;
};

union FParam_d {
    FParam_cste_d cste;
    FParam_dp_d dp;
    FParam_shear_d shear,
};

