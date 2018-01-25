/* Configuration file for sphere in channel flow */

/* DOMAIN */
#define XS                      16
#define YS                      32
#define ZS                      16
#define XWM                     6
#define YWM                     6
#define ZWM                     6
#define XBBM                    1.f
#define YBBM                    1.f
#define ZBBM                    1.f

/* DPD */
#define numberdensity           10
#define kBT                     0.0444
#define dt                      5e-4
#define adpd_b                2.6667
#define adpd_r                 2.6667
#define adpd_br                2.6667
#define gdpd_b           8.0
#define gdpd_r            12.0
#define gdpd_br           8.0
#define flu_mass                1.0
#define rbc_mass                1.0
#define solid_mass              1.0

/* FEATURES */
#define rbcs                    true
#define contactforces           true
#define ljsigma                 0.3
#define ljepsilon               0.44
#define fsiforces               true
#define tend                    1000000

#define multi_solvent           true
#define color_freq              100

/* FLOW TYPE */
#define pushsolid               true
#define pushrbc                 false

/* DUMPS */
#define dump_all_fields         true
// #define part_freq               1000
// #define field_dumps             true
// #define field_freq              1000
#define strt_dumps              true
#define strt_freq               50000
