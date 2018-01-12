struct RbcParams;
struct RbcParams_v {
    float gammaC, kBT;
    float kb, phi;
    float Cp, x0;
    float ka, kd, kv;
};

void rbc_params_ini(RbcParams **p);
void rbc_params_fin(RbcParams *p);

void rbc_params_set(RbcParams *p);

RbcParams_v rbc_params_get_view(const RbcParams *p);
