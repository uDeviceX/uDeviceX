#define numberdensity    10
#define kBT              0.0444302
#define flu_mass         1.0        /* solvent particle mass */
#define rbc_mass         0.5        /* RBC particle mass     */
#define solid_mass       0.5        /* solid particle mass   */

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
#define pushsolid        (false)
#define pushrbc          (false)
#define walls            (false)
#define RBCnv            (498)
