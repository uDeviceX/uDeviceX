/* helper functions for DPD MSD calculations */
#include "visc-aux.h"
#include "common.h"
#include "last_bit_float.h"

void set_traced_particles(int n, Particle * particles) {
  for (int i = 0; i<n; i++)
    last_bit_float::set(particles[i].u[0], false);

  for (int i = 0; i<9; i++)
    last_bit_float::set(particles[i].u[0], true);
}

std::vector<int> get_traced_list(int n, Particle * const particles) {
  std::vector<int> ilist;
  for (int i = 0; i<n; i++) {
    const bool traced = last_bit_float::get(particles[i].u[0]);
    if (traced)
      ilist.push_back(i);
  }
  return ilist;
}

void print_traced_particles(Particle * particles, int n) {
  int count = 0;
    for(int i = 0; i < n; ++i) {
        bool traced = last_bit_float::get(particles[i].u[0]);
        if (traced) count++;
    }
    printf("%d particles are traced\n", count);
}
