/* Configuration for GoogleX run */


/* DOMAIN */
#define XS                  64
#define YS                  64
#define ZS                  24

#define XWM                 6
#define YWM                 6
#define ZWM                 6

#define XBBM                1.f
#define YBBM                1.f
#define ZBBM                1.f


/* DPD */
#define numberdensity       4
#define kBT                 4.44302e-8
#define dt                  5e-4
#define rbc_mass            0.5
#define solid_mass          0.5

#define aij_solv            5
#define aij_rbc             5
#define aij_solid           5
#define aij_wall            5
#define gammadpd_solv       8.0
#define gammadpd_rbc        8.0
#define gammadpd_solid      8.0
#define gammadpd_wall       8.0



/* FEATURES */
#define rbcs                false
#define solids              false
#define contactforces       true
#define ljsigma             0.3
#define ljepsilon           0.44

#define pushflow            true
#define driving_force       0.02
#define doublepoiseuille    false
#define gamma_dot           0.0

#define walls               true
#define wall_creation       500

#define tend                3000


/* DUMPS */
#define part_dumps          true
#define field_dumps         true
#define part_freq           500
#define field_freq          500
