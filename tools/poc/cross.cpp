/* cross product c = a x b */

#include "stdio.h"

void cross(float* a, float* b, /**/ float* c) {
  enum {X, Y, Z};
  c[X] = a[Y]*b[Z]-b[Y]*a[Z];
  c[Y] = b[X]*a[Z]-a[X]*b[Z];
  c[Z] = a[X]*b[Y]-b[X]*a[Y];
}

void wavg(float* R0, float* R1, float t, /**/ float* Rt) {
  enum {X, Y, Z};
  Rt[X] = R0[X]*(1-t) + R1[X]*t;
  Rt[Y] = R0[Y]*(1-t) + R1[Y]*t;
  Rt[Z] = R0[Z]*(1-t) + R1[Z]*t;
}

/* bounce velocity of the partices
   V0: initial velocity,
   Vw: wall velocity */
void bounce_vel(float* V0, float* Vw, /**/ float* V1) {
  for (int d = 0; d < 3; d++) {
    V1[d]  = V0[d];
    
    V1[d] -=  Vw[d];
    V1[d]  = -V1[d];
    V1[d] +=  Vw[d];
  }
}

/* angular velocity and position to linear velocity */
void om2lin(float* om, float* r, /**/ float* v) {
  cross(om, r, /**/ v);
}

int main() {
  enum {X, Y, Z};  
  float a[] = {0, 1, 0};
  float b[] = {1, 0, 0};
  float c[3];

  cross(a, b, c);
  printf("c: %g %g %g\n", c[X], c[Y], c[Z]);

  float R0[] = { 1,  2,  3};
  float R1[] = {10, 20, 30};
  float Rt[3];
  wavg(R0, R1, 0.5, /**/ Rt);
  printf("Rt: %g %g %g\n", Rt[X], Rt[Y], Rt[Z]);

  float V0[] = { 1,   2,  3};
  float Vw[] = {10,  -20,  30};
  float V1[3];
  bounce_vel(V0, Vw, /**/ V1);
  printf("V1: %g %g %g\n", V1[X], V1[Y], V1[Z]);

  float om[] = { 0,   0,  10};
  float r[] =  { 0,   1,   0};
  float v[3];
  om2lin(om, r, v);
  printf("v: %g %g %g\n", v[X], v[Y], v[Z]);
}
