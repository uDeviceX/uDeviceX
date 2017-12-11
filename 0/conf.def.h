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
#define numberdensity           10
#define kBT                     0.00444302
#define dt                      1e-3
#define flu_mass                1.0
#define rbc_mass                0.5
#define solid_mass              1.0

#define adpd_b         2.6
#define adpd_r         2.6
#define adpd_br        2.6

#define gdpd_b         1.0
#define gdpd_r         5.0
#define gdpd_br        3.0


/* FEATURES */
#define rbcs                    false
#define multi_solvent           true
#define color_freq              500
#define contactforces           true
#define ljsigma                 0.3
#define ljepsilon               0.44
#define fsiforces               true
#define walls                   true
#define wall_creation           1000
#define tend                    1000000

/* FLOW TYPE */
#define pushflow                true
#define pushrbc                 true
#define driving_force           0.001

/* DUMPS */
#define dump_all_fields         true
#define part_freq               20000
#define field_dumps             true
#define field_freq              200000
#define strt_dumps              true
#define strt_freq               200000
