float x0 = RBCx0, p = RBCp, ka = RBCka, kb = RBCkb,
  kd = RBCkd, kv = RBCkv,
  gammaC = RBCgammaC, totArea0 = RBCtotArea,
  totVolume0 = RBCtotVolume, nvertices = RBCnv;

float RBCscale = 1.0/rc;
float ll =  1.0/RBCscale;

float kBT2D3D = 1;
float phi = 6.97 / 180.0 * M_PI; /* theta_0 */

float sinTheta0 = sin(phi);
float cosTheta0 = cos(phi);
float kbT = 0.1 * kBT2D3D;
float mpow = 2; /* WLC-POW */

/* units conversion: Fedosov -> uDeviceX */
kv = kv * ll;
p = p / ll;
totArea0 = totArea0 / (ll * ll);
kb = kb / (ll * ll);
kbT = kbT / (ll * ll);
totVolume0 = totVolume0 / (ll * ll * ll);

// derived parameters
float Area0 = totArea0 / (2.0 * nvertices - 4.);
float l0 = sqrt(Area0 * 4.0 / sqrt(3.0));
float lmax = l0 / x0;
float gammaT = 3.0 * gammaC;
float kbToverp = kbT / p;
float sint0kb = sinTheta0 * kb;
float cost0kb = cosTheta0 * kb;
float kp =
  (kbT * x0 * (4 * x0 * x0 - 9 * x0 + 6) * l0 * l0) /
  (4 * p * (x0 - 1) * (x0 - 1));

/* to simplify further computations */
ka = ka / totArea0;
kv = kv / (6 * totVolume0);
