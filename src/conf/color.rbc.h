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
#define aij_solv                2.6667
#define aij_rbc                 2.6667
#define aij_solid               2.6667
#define aij_wall                2.6667
#define gammadpd_solv           8.0
#define gammadpd_rbc            12.0
#define gammadpd_solid          8.0
#define gammadpd_wall           8.0
#define dpd_mass                1.0
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
#define FORCE_PAR_A           0.2

/* DUMPS */
#define dump_all_fields         true
#define part_freq               1000
#define field_dumps             true
#define field_freq              1000
#define strt_dumps              true
#define strt_freq               50000
