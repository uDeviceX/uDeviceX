#include <stdio.h>
#include <assert.h>
#include <conf.h>
#include "inc/conf.h"

#include "mpi/glb.h"
#include "glb/imp.h"

#include "utils/cc.h"
#include "utils/kl.h"
#include "inc/def.h"
#include "inc/dev.h"
#include "d/api.h"

#include "forces/type.h"
#include "forces/use.h"
#include "forces/imp.h"

namespace dev {
__global__ void main(forces::Pa a, forces::Pa b) {
    forces::Fo f;
    float rnd;
    rnd = 0.0;
    forces::gen(a, b, rnd, /**/ &f);
    printf("fo: %g %g %g\n", f.x, f.y, f.z);
}

}

void ini_pa(float x, float y, float z,
           float vx, float vy, float vz,
           int k, int c, /**/ forces::Pa *a) {
    a->x  = x;  a->y  = y;  a->z =  z;
    a->vx = vx; a->vy = y; a->vz = vz;    
    a->kind = k;
    a->color = c;
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    
    forces::Pa a;
    forces::Pa b;
    ini_pa(0,0,0, 0,0,0, SOLID_KIND, BLUE_COLOR, /**/ &a);
    ini_pa(0.1,0.1,0.1, 0,0,0, SOLID_KIND, BLUE_COLOR, /**/ &b);
    
    KL(dev::main, (1, 1), (a, b));
    dSync();
    m::fin();
}
