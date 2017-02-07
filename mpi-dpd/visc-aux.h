#pragma once

/* helper functions for DPD MSD calculations */
#include <vector>

class Particle;

/* set traced particle using last_bit_float */
void             set_traced_particles(int n, Particle * particle);

/* get a list on indices of traced particles
   ilist[0], ilist[1] is an index of traced particle in the array  */
std::vector<int> get_traced_list(int n, Particle * particles);

void print_traced_particles(Particle * particles, int n);

typedef std::vector<float> TVec;
void hello_a(TVec& sol_xx, TVec& sol_yy, TVec& sol_zz,
	     TVec& rbc_xx, TVec& rbc_yy, TVec& rbc_zz);

__device__ float3 compute_dpd_force_traced(int type1, int type2,
    float3 pos1, float3 pos2, float3 vel1, float3 vel2, float myrandnr);
