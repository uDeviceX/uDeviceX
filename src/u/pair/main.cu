#include <stdio.h>
#include <assert.h>
#include <conf.h>
#include "inc/conf.h"

#include "msg.h"
#include "mpi/glb.h"
#include "glb/imp.h"

#include "utils/cc.h"
#include "utils/kl.h"
#include "inc/def.h"

#include "forces/type.h"
#include "forces/use.h"
#include "forces/imp.h"

namespace dev {
__global__ void main() {
    forces::Pa a;
    forces::Pa b;
    forces::Fo f;
    float rnd;

    a.x = a.y = a.z = a.vx = a.vy = a.vz = 0.0;
    a.kind = SOLID_KIND;
    a.color = RED_COLOR;    

    b.x = b.y = b.z = b.vx = b.vy = b.vz = 0.0;
    b.kind = SOLID_KIND;
    a.color = RED_COLOR;

    rnd = 0.0;

    forces::gen(a, b, rnd, /**/ &f);
    printf("fo: %g %g %g\n", f.x, f.y, f.z);
}

}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    MSG("mpi size: %d", m::size);
    MSG("Hello world!");
    KL(dev::main, (1, 1), ());
    m::fin();
}
