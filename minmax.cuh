#pragma once

struct Particle;
struct float3;

void minmax(const Particle *pp, int nv, int nobj, float3 *lo, float3 *hi);
