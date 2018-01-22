#define XS   16
#define YS   32
#define ZS   16
#define XWM     6
#define YWM     6
#define ZWM     6

#define XBBM 1.f
#define YBBM 1.f
#define ZBBM 1.f

#define numberdensity    10
#define kBT              0.0444302
#define dt               5e-4       /* timestep */
#define flu_mass         1.0        /* solvent particle mass */
#define rbc_mass         0.5        /* RBC particle mass     */
#define solid_mass       0.5        /* solid particle mass   */
// #define WVEL_PAR_A        5.0        /* shear rate */

#define adpd_b         2
#define adpd_r          2
#define adpd_br         10

#define gdpd_b    1
#define gdpd_r     100
#define gdpd_br    10

#define ljsigma          0.3        /* RBC-RBC contact LJ interaction parameters */
#define ljepsilon        0.44

#define global_ids       (false)
#define multi_solvent    (true)
#define color_freq       (1000)

#define fsiforces        (true)
#define contactforces    (false)
#define strt_dumps       (false)
#define strt_freq        (5000)
#define field_dumps      (false)
#define field_freq       (2000)
#define part_dumps       (false)
#define part_freq        (1000)
#define pushsolid        (false)
#define pushrbc          (false)
#define tend             (50)
#define wall_creation    (5000)
#define walls            (false)
#define RBCnv            (498)
