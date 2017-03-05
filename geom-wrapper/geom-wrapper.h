#ifndef GEOM_WRAPPER_IOTAGS_H
#define GEOM_WRAPPER_IOTAGS_H

/* `faces' are in one array
   nb: number of vertices in one RBC
   nf: number of faces
   f0[0] f1[0] f2[0]   f0[1] f1[1] ... */
void iotags_init(long nb_, long  nf_, int* faces);

/* set domain sizes and periodic flags */
void iotags_domain(float xl, float yl, float zl,
                   float xh, float yh, float zh,
                   int pbcx_, int pbcy_, int pbcz_);

/* Sets in-out flags for solvent particles:
   -1: outside, [0:nrbc-1] id of the particle it belongs to
Note:
Modifies sol_[xx|yy|zz] */
void iotags_all(long  nrbc , float* rbc_xx, float* rbc_yy, float* rbc_zz,
                long  nsol_, /* modified */ float* sol_xx, float* sol_yy, float* sol_zz,
                /* output   */   int* iotags);

/* dealocate ff1, ff2, ff3 */
void iotags_fin();

#endif
