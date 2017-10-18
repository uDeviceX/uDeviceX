/* Part I (conf.base.h) */

/* DOMAIN */
#define XS                      64
#define YS                      52
#define ZS                      56
#define XWM                     6
#define YWM                     6
#define ZWM                     6
#define XBBM                    1.f
#define YBBM                    1.f
#define ZBBM                    1.f

/* DPD */
#define numberdensity           4
#define kBT                     0.0444
#define dt                      1e-3
#define dpd_mass                1.0
#define rbc_mass                0.5
#define solid_mass              1.0

#define adpd_b         2.6
#define adpd_r         2.6
#define adpd_br        2.6

#define gdpd_b         8.0
#define gdpd_r         8.0
#define gdpd_br        8.0


/* FEATURES */
#define rbcs                    false
#define multi_solvent           false
#define color_freq              500
#define contactforces           true
#define ljsigma                 0.3
#define ljepsilon               0.44
#define fsiforces               true
#define walls                   true
#define wall_creation           1000
#define tend                    1000000

/* DEBUG */
#define TE_TRACE

/* FLOW TYPE */
#define pushflow                true
#define pushrbc                 false
#define driving_force           0.001

/* DUMPS */
#define dump_all_fields         true
#define part_freq               2000
#define field_dumps             true
#define field_freq              2000
#define strt_dumps              true
#define strt_freq               100000
