#include <mpi.h>
#include "conf.h"
#include "conf.default.h"
#include "common.h"
#include "m.h"
#include "glb.h"

#define ndim 3
#define X 0
#define Y 1
#define Z 2

/* global variables visible for every kernel */
namespace glb {
__constant__ float r0[ndim];
__constant__ float lg[ndim];

void sim() {
    /* all coordinates are relative to the center of the sub-domain;
       Example: (dims[X] = 3, `XS' is sub-domain size):
       |            |             |             |
       -XS/2          XS/2        3XS/2         5XS/2
       coords[X]=0   coords[X]=1   coords[X]=2
    */

    float r0_h[3]; /* the center of the domain in sub-domain
                      coordinates; to go to domain coordinates (`rg')
                      from sub-domain coordinates (`r'): rg = r - r0
                   */
    r0_h[X] = XS*(m::dims[X]-2*m::coords[X]-1)/2;
    r0_h[Y] = YS*(m::dims[Y]-2*m::coords[Y]-1)/2;
    r0_h[Z] = ZS*(m::dims[Z]-2*m::coords[Z]-1)/2;
    cudaMemcpyToSymbol(r0, r0_h, ndim*sizeof(float));

    float lg_h[3]; /* domain size */
	lg_h[X] = m::dims[X] * XS;
	lg_h[Y] = m::dims[Y] * YS;
	lg_h[Z] = m::dims[Z] * ZS;
    cudaMemcpyToSymbol(lg, lg_h, ndim*sizeof(float));
}
}
#undef X
#undef Y
#undef Z
#undef ndim
