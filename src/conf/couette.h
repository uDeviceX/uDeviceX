#define XS   16
#define YS   16
#define ZS   16
#define XWM     6
#define YWM     6
#define ZWM     6

#define XBBM 1.f
#define YBBM 1.f
#define ZBBM 1.f

#define numberdensity    10
#define kBT              1e-7
#define dt               5e-4  /* timestep */
#define flu_mass         1.0        /* solvent particle mass */
#define rbc_mass         0.5        /* RBC particle mass     */
#define solid_mass       0.5        /* solid particle mass   */

#define adpd_b         2.667  /* aij for the solvent  */
#define adpd_r          2.667  /* aij for the RBC membrane */
#define adpd_br         2.667  /* aij for the wall */

#define gdpd_b    8.0  /* gamma for the solvent */
#define gdpd_r     8.0  /* gamma for the RBC membrane */
#define gdpd_br    8.0  /* gamma for the wall */

#define global_ids       (false)
#define multi_solvent    (false)

#define ljsigma          0.3   /* RBC-RBC contact LJ interaction parameters */
#define ljepsilon        0.444

#define fsiforces        (true)
#define contactforces    (false)
#define strt_dumps       (false)
#define strt_freq        (5000)
#define pushsolid        (false)
#define pushrbc          (false)
#define tend             (50)
#define wall_creation    (5000)
#define walls            (false)
#define RBCnv            (498)
