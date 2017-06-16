namespace k_rbc {
#define sq(a) ((a)*(a))
#define abscross2(a, b)                         \
    (sq((a).y*(b).z - (a).z*(b).y) +            \
     sq((a).z*(b).x - (a).x*(b).z) +            \
     sq((a).x*(b).y - (a).y*(b).x))
#define abscross(a, b) sqrtf(abscross2(a, b)) /* |a x b| */

#define cross(a, b) make_float3                 \
    ((a).y*(b).z - (a).z*(b).y,                 \
     (a).z*(b).x - (a).x*(b).z,                 \
     (a).x*(b).y - (a).y*(b).x)

__DF__ float3 tri(float3 v1, float3 v2,
		     float3 v3, float area,
		     float volume) {
#include "params/rbc.inc0.h"
    float Ak, A0, n_2, coefArea, coeffVol,
	r, xx, IbforceI_wcl, kp, IbforceI_pow, ka0, kv0, x0, l0, lmax,
	kbToverp;

    float3 x21 = v2 - v1, x32 = v3 - v2, x31 = v3 - v1;
    float3 nn = cross(x21, x31); /* normal */

    Ak = 0.5 * sqrtf(dot(nn, nn));

    A0 = RBCtotArea / (2.0 * RBCnv - 4.);
    n_2 = 1.0 / Ak;
    ka0 = RBCka / RBCtotArea;
    coefArea =
	-0.25f * (ka0 * (area - RBCtotArea) * n_2) -
	RBCkd * (Ak - A0) / (4. * A0 * Ak);

    kv0 = RBCkv / (6.0 * RBCtotVolume);
    coeffVol = kv0 * (volume - RBCtotVolume);
    float3 addFArea = coefArea * cross(nn, x32);
    float3 addFVolume = coeffVol * cross(v3, v2);

    r = sqrtf(dot(x21, x21));
    r = r < 0.0001f ? 0.0001f : r;
    l0 = sqrt(A0 * 4.0 / sqrt(3.0));
    lmax = l0 / RBCx0;
    xx = r / lmax;

    kbToverp = RBCkbT / RBCp;
    IbforceI_wcl =
	    kbToverp * (0.25f / ((1.0f - xx) * (1.0f - xx)) - 0.25f + xx) /
	    r;

    x0 = RBCx0;
    kp =
	    (RBCkbT * x0 * (4 * x0 * x0 - 9 * x0 + 6) * l0 * l0) /
	    (4 * RBCp * (x0 - 1) * (x0 - 1));
    IbforceI_pow = -kp / powf(r, RBCmpow) / r;

    return addFArea + addFVolume + (IbforceI_wcl + IbforceI_pow) * x21;
}

__DF__ float3 visc(float3 v1, float3 v2,
					 float3 u1, float3 u2) {
    float3 du = u2 - u1, dr = v1 - v2;
    float gammaC = RBCgammaC, gammaT = 3.0 * RBCgammaC;

    return gammaT                             * du +
	   gammaC * dot(du, dr) / dot(dr, dr) * dr;
}

template <int update>
__DF__ float3 dihedral(float3 v1, float3 v2, float3 v3,
					     float3 v4) {
    float overIksiI, overIdzeI, cosTheta, IsinThetaI2, sinTheta_1,
	beta, b11, b12, phi, sint0kb, cost0kb;

    float3 ksi = cross(v1 - v2, v1 - v3), dze = cross(v3 - v4, v2 - v4);
    overIksiI = rsqrtf(dot(ksi, ksi));
    overIdzeI = rsqrtf(dot(dze, dze));

    cosTheta = dot(ksi, dze) * overIksiI * overIdzeI;
    IsinThetaI2 = 1.0f - cosTheta * cosTheta;

    sinTheta_1 = copysignf
	(rsqrtf(max(IsinThetaI2, 1.0e-6f)),
	 dot(ksi - dze, v4 - v1)); // ">" because the normals look inside

    phi = RBCphi / 180.0 * M_PI;
    sint0kb = sin(phi) * RBCkb;
    cost0kb = cos(phi) * RBCkb;
    beta = cost0kb - cosTheta * sint0kb * sinTheta_1;

    b11 = -beta *  cosTheta * overIksiI * overIksiI;
    b12 =  beta * overIksiI * overIdzeI;

    if (update == 1) {
	return b11 * cross(ksi, v3 - v2) + b12 * cross(dze, v3 - v2);
    } else if (update == 2) {
	float b22 = -beta * cosTheta * overIdzeI * overIdzeI;
	return  b11 *  cross(ksi, v1 - v3) +
	    b12 * (cross(ksi, v3 - v4) + cross(dze, v1 - v3)) +
	    b22 *  cross(dze, v3 - v4);
    } else
    return make_float3(0, 0, 0);
}

__DF__ float area0(float3 v0, float3 v1, float3 v2) { return 0.5f * abscross(v1 - v0, v2 - v0); }
__DF__ float volume0(float3 v0, float3 v1, float3 v2) {
  return \
    0.1666666667f *
    ((v0.x*v1.y-v0.y*v1.x)*v2.z +
     (v0.z*v1.x-v0.x*v1.z)*v2.y +
     (v0.y*v1.z-v0.z*v1.y)*v2.x);
}
#undef sq
#undef abscross2
#undef abscross
} /* namespace k_rbc */
