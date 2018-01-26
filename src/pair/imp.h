struct PairParams;
struct Config;

void pair_ini(PairParams **);
void pair_fin(PairParams *);

void pair_set_lj(float sigma, float eps, PairParams *p);
void pair_set_dpd(int ncol, const float a[], const float g[], const float s[], PairParams *p);

void pair_set_conf(const Config *, PairParams *);

