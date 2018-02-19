#include <stdio.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"
#include "utils/msg.h"
#include "inc/type.h"
#include "utils/kl.h"
#include "utils/cc.h"
#include "d/api.h"
#include "inc/dev.h"
#include "coords/type.h"
#include "coords/imp.h"

#include "flux.h"

namespace color_dev {
__global__ void linear_flux(int3 L, int dir, int color, int n, const Particle *pp, int *cc) {
    int i;
    Particle p;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    p = pp[i];
    const int HL[] = {L.x/2, L.y/2, L.z/2};

    if (p.r[dir] >= HL[dir])
        cc[i] = color;
}
}

void color_linear_flux(const Coords *coords, int3 L, int dir, int color, int n, const Particle *pp, int *cc) {
    assert(dir >= 0 && dir <= 2);
    if (is_end(coords, dir))
        KL(color_dev::linear_flux, (k_cnf(n)), (L, dir, color, n, pp, cc));
}
