#ifndef GEOM_WRAPPER_IOTAGS_H
#define GEOM_WRAPPER_IOTAGS_H

void iotags_init(long nb_, long  nf_, int*   rbc_f1, int*   rbc_f2, int* rbc_f3);
void iotags_init_file(const char* fn);
void iotags_domain(float Lx_, float Ly_, float Lz_,
		   int pbcx_, int pbcy_, int pbcz_);
void iotags_all(long  nrbc , float* rbc_xx, float* rbc_yy, float* rbc_zz,
		long  nsol_, float* sol_xx, float* sol_yy, float* sol_zz,
		int* iotags);

#endif
