#pragma once

/* helper functions for DPD MSD calculations */
#include <vector>

class Particle;

/* set traced particle using last_bit_float */
void             set_traced_particles(int n, Particle * particle);

/* get a list on indices of traced particles
   ilist[0], ilist[1] is an index of traced particle in the array  */
std::vector<int> get_traced_list(int n, Particle * particles);
