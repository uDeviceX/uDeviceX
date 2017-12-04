struct Params {
    float3 o;   /* center of cylinder */
    float R, H; /* radius, hight      */
    float th0, dth;
};

struct VParams {
    float u;
    bool poiseuille;
};
