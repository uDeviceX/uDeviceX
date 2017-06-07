#define XS   16
#define YS   16
#define ZS   16
#define XMARGIN_WALL     6
#define YMARGIN_WALL     6
#define ZMARGIN_WALL     6

#define XMARGIN_BB 1.f
#define YMARGIN_BB 1.f
#define ZMARGIN_BB 1.f

#define  numberdensity    10
#define kBT              1e-7
#define  dt               5e-4  /* timestep */
#define  rbc_mass         0.5   /* ratio of RBC particle mass to solvent particle mass */
#define gamma_dot        5.0       /* shear rate */
#define driving_force    3.333e-2  /* flow acceleration for Poiseuille setup */

#define aij_out          2.667  /* aij for the solvent outside the RBC */
#define aij_in           2.667  /* aij for the solvent inside the RBC */
#define aij_rbc          2.667  /* aij for the RBC membrane */
#define aij_solid        2.667  /* aij for the solid */
#define aij_wall         2.667  /* aij for the wall */

#define gammadpd_out     8.0  /* gamma for the solvent outside the RBC */
#define gammadpd_in      8.0  /* gamma for the solvent inside the RBC */
#define gammadpd_rbc     8.0  /* gamma for the RBC membrane */
#define gammadpd_solid   8.0        /* gamma for the solid */
#define gammadpd_wall    8.0  /* gamma for the wall */

#define ljsigma          0.3   /* RBC-RBC contact LJ interaction parameters */
#define ljepsilon        0.444
