/* helper functions for DPD MSD calculations */
#include "visc-aux.h"
#include "common.h"
#include "last_bit_float.h"

extern float RBCscale;

bool is_inside_rbc(float x, float y, float z, float th) {
    x *= th; y *= th; z *= th;

    float a0 = 0.0518, a1 = 2.0026, a2 = -4.491;
    float D0 = 7.82;

    float rho = (x*x+y*y)/(D0*D0);
    float s = 1-4*rho;
    if (s < 0)
        return false;

    float zrbc = D0*sqrt(s)*(a0 + a1*rho + a2*pow(rho,2));

    return z > -zrbc && z < zrbc;
}

void set_traced_particles(int n, Particle * particles) {
  for (int i = 0; i<n; i++)
    last_bit_float::set(particles[i].u[0], false);

  for (int i = 0; i<n; i++) {
    float x = particles[i].x[0];
    float y = particles[i].x[1];
    float z = particles[i].x[2];
    if (is_inside_rbc(x, y, z, 1.1/RBCscale))
      last_bit_float::set(particles[i].u[0], true);
  }
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
