struct PairParams;
struct Config;

struct PairDPD;
struct PairDPDC;
struct PairDPDCM;
struct PairDPDLJ;

void pair_ini(PairParams **);
void pair_fin(PairParams *);

void pair_set_lj(float sigma, float eps, PairParams *p);
void pair_set_dpd(int ncol, const float a[], const float g[], const float s[], PairParams *p);

void pair_set_conf(const Config *, PairParams *);

void pair_get_view_dpd(const PairParams*, PairDPD*);
void pair_get_view_dpd_color(const PairParams*, PairDPDC*);
void pair_get_view_dpd_mirrored(const PairParams*, PairDPDCM*);
void pair_get_view_dpd_lj(const PairParams*, PairDPDLJ*);
