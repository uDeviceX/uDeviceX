/* Part I (conf.base.h) */

/* DOMAIN */
#define XS                  4
#define YS                  4
#define ZS                  4
#define XWM                 6
#define YWM                 6
#define ZWM                 6
#define XBBM                1.f
#define YBBM                1.f
#define ZBBM                1.f

/* DPD */
#define numberdensity       1
#define kBT                 1.e-8
#define dt                  0.0005
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
#define tend                100

/* DEBUG */
//#define KL_NONE
//#define CC_PEEK
//#define TE_TRACE
//#define FORCE0

/* DUMPS */
#define part_freq           10000
#define field_dumps         false
#define field_freq          10000
#define strt_dumps          false
#define strt_freq           10000
