/* DPD */
#define numberdensity       1
#define kBT                 1.e-8
#define adpd_b             (75.0/numberdensity)
#define adpd_r             (75.0/numberdensity)
#define adpd_br            (75.0/numberdensity)
#define gdpd_b        40.0
#define gdpd_r        40.0
#define gdpd_br       40.0
#define flu_mass            1.0
#define rbc_mass            1.0
#define solid_mass          1.0

/* FEATURES */
#define ljsigma             0.3
#define ljepsilon           0.44
#define solids              false
#define sbounce_back        true
#define fsiforces           true
#define walls               false
#define wall_creation       1000
//#define tend                100

/* DUMPS */
#define strt_dumps          false
#define strt_freq           10000
