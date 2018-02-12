struct PairParams;
struct Config;

struct PairDPD;
struct PairDPDC;
struct PairDPDCM;
struct PairDPDLJ;

// tag::mem[]
void pair_ini(PairParams **);
void pair_fin(PairParams *);
// end::mem[]

// tag::set[]
void pair_set_lj(float sigma, float eps, PairParams *p);
void pair_set_dpd(int ncol, const float a[], const float g[], PairParams *p);
// end::set[]

// tag::cnf[]
void pair_set_conf(const Config *, const char *name_space, PairParams *);
// end::cnf[]

// tag::upd[]
void pair_compute_dpd_sigma(float kBT0, float dt, PairParams *p);
// end::upd[]

// tag::view[]
void pair_get_view_dpd(const PairParams*, PairDPD*);
void pair_get_view_dpd_color(const PairParams*, PairDPDC*);
void pair_get_view_dpd_mirrored(const PairParams*, PairDPDCM*);
void pair_get_view_dpd_lj(const PairParams*, PairDPDLJ*);
// end::view[]

