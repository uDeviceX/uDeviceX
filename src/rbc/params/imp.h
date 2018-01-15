struct Config;
struct RbcParams;
struct RbcParams_v {
    float gammaC, kBT;
    float kb, phi;
    float Cp, x0;
    float ka, kd, kv;
};

void rbc_params_ini(RbcParams **p);
void rbc_params_fin(RbcParams *p);

void rbc_params_set_fluct(float gammaC, float kBT, RbcParams *p);
void rbc_params_set_bending(float kb, float phi, RbcParams *p);
void rbc_params_set_spring(float Cp, float x0, RbcParams *p);
void rbc_params_set_area_volume(float ka, float kd, float kv, RbcParams *p);

void rbc_params_set_conf(const Config *c, RbcParams *p);

RbcParams_v rbc_params_get_view(const RbcParams *p);
