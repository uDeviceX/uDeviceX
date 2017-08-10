#include <conf.h>
#include "inc/conf.h"
#include "m.h"
#include "glb.h"

/* global variables visible for every kernel */
namespace glb {
__constant__ float r0[3];
__constant__ float lg[3];

void sim() {
    enum {X, Y, Z};
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
    cudaMemcpyToSymbol(r0, r0_h, 3*sizeof(float));

    float lg_h[3]; /* domain size */
	lg_h[X] = m::dims[X] * XS;
	lg_h[Y] = m::dims[Y] * YS;
	lg_h[Z] = m::dims[Z] * ZS;
    cudaMemcpyToSymbol(lg, lg_h, 3*sizeof(float));
}
}
