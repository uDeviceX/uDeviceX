struct PairParams;
struct Config;

void pair_ini(PairParams **);
void pair_fin(PairParams *);

void pair_set_conf(const Config *, PairParams *);

