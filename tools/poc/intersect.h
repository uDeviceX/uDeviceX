bool intersect(float* P0, float* P1, float R, float *t)  {
  const float eps = 1e-8;
  enum {X, Y, Z};

  float P0x = P0[X], P0y = P0[Y], P0z = P0[Z];
  float P1x = P1[X], P1y = P1[Y], P1z = P1[Z];

  P0x /= R; P0y /= R; P0z /= R;
  P1x /= R; P1y /= R; P1z /= R;

  float a, b, c, D, sqD, t0, t1;
  a = pow(P1z-P0z,2)+pow(P1y-P0y,2)+pow(P1x-P0x,2);
  if (a < eps) return false;

  b = 2*P0z*(P1z-P0z)+2*P0y*(P1y-P0y)+2*P0x*(P1x-P0x);
  c = pow(P0z,2)+pow(P0y,2)+pow(P0x,2)-1;

  D = b*b - 4*a*c;
  if (D < 0) return false;

  sqD = sqrt(D);
  t0 = (-b - sqD)/(2*a);
  if (t0 > 0 && t0 < 1) {*t = t0; return true;}

  t1 = (-b + sqD)/(2*a);
  if (t1 > 0 && t1 < 1) {*t = t1; return true;}

  return false;
}
