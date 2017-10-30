#include <assert.h>

#include <conf.h>
#include "inc/conf.h"
#include "inc/type.h"
#include "utils/kl.h"
#include "inc/dev.h"
#include "mpi/glb.h"

#include "flux.h"

namespace recolor {
namespace dev {
__global__ void flux(int dir, int color, int n, const Particle *pp, int *cc) {
    int i;
    Particle p;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    
    p = pp[i];
    const int L[] = {XS/2, YS/2, ZS/2};

    if (p.r[dir] >= L[dir])
        cc[i] = color;
}
} // dev

void flux(int dir, int color, int n, const Particle *pp, int *cc) {
    assert(dir >= 0 && dir <= 2);

    if (m::coords[dir] == m::dims[dir])
        KL(dev::flux, (k_cnf(n)), (dir, color, n, pp, cc));
}
} // recolor
