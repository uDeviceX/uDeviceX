#define rc                1.5  /* cutoff radius */

#define XSIZE_SUBDOMAIN   16
#define YSIZE_SUBDOMAIN   16
#define ZSIZE_SUBDOMAIN   16
#define XMARGIN_WALL      6
#define YMARGIN_WALL      6
#define ZMARGIN_WALL      6

#define _numberdensity    3
#define _kBT              1e-7
#define  dt               5e-4  /* timestep */
#define _rbc_mass         0.5   /* ratio of RBC particle mass to solvent particle mass */
#define _gamma_dot        0.0   /* shear rate */
#define _hydrostatic_a    2.5   /* flow acceleration for Poiseuille setup */

#define _aij_out          4.0  /* aij for the solvent outside the RBC */
#define _aij_in           4.0  /* aij for the solvent inside the RBC */
#define _aij_rbc          4.0  /* aij for the RBC membrane */
#define _aij_wall         4.0  /* aij for the wall */

#define _gammadpd_out     8.0  /* gamma for the solvent outside the RBC */
#define _gammadpd_in      8.0  /* gamma for the solvent inside the RBC */
#define _gammadpd_rbc     8.0  /* gamma for the RBC membrane */
#define _gammadpd_wall    8.0  /* gamma for the wall */

#define _ljsigma          0.3   /* RBC-RBC contact LJ interaction parameters */
#define _ljepsilon        1.0
