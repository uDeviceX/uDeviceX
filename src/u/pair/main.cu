#include <stdio.h>
#include <assert.h>
#include <stdio.h>

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

typedef forces::Fo Fo;
typedef forces::Pa Pa;

namespace dev {
__global__ void main(Pa a, Pa b, float rnd) {
    Fo f;
    forces::gen(a, b, rnd, /**/ &f);
    printf("%g %g %g\n", f.x, f.y, f.z);
}
} /* namespace */

void ini_pa(float x, float y, float z,
            float vx, float vy, float vz,
            int k, int c, /**/ Pa *a) {
    a->x  = x;  a->y  = y;  a->z =  z;
    a->vx = vx; a->vy = y; a->vz = vz;
    a->kind = k;
    a->color = c;
}

void pair(Pa a, Pa b, float rnd) {
    KL(dev::main, (1, 1), (a, b, rnd));
    dSync();
}

void write_pa(Pa *a) {
    printf("[ %g %g %g ] [ %g %g %g ] [kc: %d %d]\n",
           a->x, a->y, a->z, a->vx, a->vy, a->vz, a->kind, a->color);
}

void read_pa(Pa *a) {
    scanf("%f %f %f   %f %f %f   %d %d",
          &a->x, &a->y, &a->z, &a->vx, &a->vy, &a->vz,
          &a->kind, &a->color);
    write_pa(a);
}

void main0() {
    Pa a, b;
    float rnd;

    read_pa(&a);
    ini_pa(0.1,0.1,0.1, 0,0,0, SOLID_KIND, BLUE_COLOR, /**/ &b);
    rnd = 0;
    pair(a, b, rnd);
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main0();
    m::fin();
}
