struct Config;
struct RbcParams;
struct RbcParams_v {
    float gammaC, gammaT, kBT0;
    float kb, phi;
    float ks, x0, mpow;
    float ka, kd, kv;
};

void rbc_params_ini(RbcParams **);
void rbc_params_fin(RbcParams *);

void rbc_params_set_fluct(float gammaC, float gammaT, float kBT0, RbcParams *);
void rbc_params_set_bending(float kb, float phi, RbcParams *);
void rbc_params_set_spring(float ks, float x0, float mpow , RbcParams *);
void rbc_params_set_area_volume(float ka, float kd, float kv, RbcParams *);

void rbc_params_set_conf(const Config *c, RbcParams *);

RbcParams_v rbc_params_get_view(const RbcParams *);
