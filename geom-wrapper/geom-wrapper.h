#ifndef GEOM_WRAPPER_IOTAGS_H
#define GEOM_WRAPPER_IOTAGS_H

/* call one of the _init_ functions:
   sets
   nb: number of vertices in one RBC
   nf: number of faces
   rbc_f1, rbc_f2, rbc_f2: zero based indexes for the faces */
void iotags_init(long nb_, long  nf_, int*   rbc_f1, int*   rbc_f2, int* rbc_f3);
void iotags_init_file(const char* fn);

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
#endif
