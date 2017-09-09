#include <conf.h>
#include "inc/conf.h"
#include "mpi/glb.h"
#include "d/api.h"
#include "glb.h"

/* globals for all kernels */
namespace glb {
__constant__ float r0[3];

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
    int *c = m::coords;
    int *d = m::dims;
    r0_h[X] = XS*(d[X]-2*c[X]-1)/2;
    r0_h[Y] = YS*(d[Y]-2*c[Y]-1)/2;
    r0_h[Z] = ZS*(d[Z]-2*c[Z]-1)/2;
    d::MemcpyToSymbol(r0, r0_h, 3*sizeof(*r0_h));
}
}
