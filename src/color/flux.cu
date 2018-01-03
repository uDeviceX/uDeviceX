#include <stdio.h>
#include <assert.h>

#include <conf.h>
#include "inc/conf.h"
#include "msg.h"
#include "inc/type.h"
#include "utils/kl.h"
#include "utils/cc.h"
#include "d/api.h"
#include "inc/dev.h"
#include "mpi/glb.h"
#include "glob/type.h"
#include "glob/imp.h"

#include "flux.h"

namespace recolor {
namespace dev {
__global__ void linear_flux(int dir, int color, int n, const Particle *pp, int *cc) {
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

void linear_flux(Coords coords, int dir, int color, int n, const Particle *pp, int *cc) {
    assert(dir >= 0 && dir <= 2);
    assert(multi_solvent);
        
    if (is_end(coords, dir))
        KL(dev::linear_flux, (k_cnf(n)), (dir, color, n, pp, cc));
}
} // recolor
