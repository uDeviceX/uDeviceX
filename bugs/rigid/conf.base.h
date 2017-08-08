/* DOMAIN */
#define XWM                 6
#define YWM                 6
#define ZWM                 6
#define XBBM                1.f
#define YBBM                1.f
#define ZBBM                1.f

/* DPD */
#define numberdensity       10
#define kBT                 0.0444
#define dt                  5.e-4
#define aij_solv            2.667
#define aij_rbc             2.667
#define aij_solid           2.667
#define aij_wall            2.667
#define gammadpd_solv       8.0
#define gammadpd_rbc        8.0
#define gammadpd_solid      8.0
#define gammadpd_wall       8.0
#define dpd_mass            1.0
#define rbc_mass            1.0
#define solid_mass          1.0

/* FEATURES */
#define solids              true
#define sbounce_back        true
#define RBCnv               498
#define ljsigma             0.3
#define ljepsilon           0.44
#define fsiforces           true
#define contactforces       false
#define multi_solvent       false
#define global_ids          false
#define walls               true
#define wall_creation       1000
#define tend                1000

/* FLOW TYPE */
#define gamma_dot           0.0
#define doublepoiseuille    false
#define pushflow            true
#define pushrbc             false
#define pushsolid           false
#define driving_force       0.05

/* DUMPS */
#define strt_dumps          false
#define strt_freq           10000
#define field_dumps         true
#define field_freq          1000
#define part_dumps          true
#define part_freq           1000
