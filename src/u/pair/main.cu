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

void pair(Pa a, Pa b, float rnd) {
    KL(dev::main, (1, 1), (a, b, rnd));
    dSync();
}

void write_pa(Pa *a) {
    printf("[ %g %g %g ] [ %g %g %g ] [kc: %d %d]\n",
           a->x, a->y, a->z, a->vx, a->vy, a->vz, a->kind, a->color);
}

void read_pa0(const char *s, Pa *a) {
    sscanf(s,
           "%f %f %f   %f %f %f   %d %d",
           &a->x, &a->y, &a->z, &a->vx, &a->vy, &a->vz,
           &a->kind, &a->color);
    write_pa(a);
}

enum {OK, END, FAIL};
int read_pa(Pa *a) {
    char s[BUFSIZ];
    if (fgets(s, BUFSIZ - 1, stdin) == NULL) return END;
    read_pa0(s, /**/ a);
    return OK;
}

void main0() {
    Pa a, b;
    float rnd;
    rnd = 0;

    for (;;) {
        if (read_pa(&a) == END) break;
        if (read_pa(&b) == END) break;
        pair(a, b, rnd);
    }
}

int main(int argc, char **argv) {
    m::ini(argc, argv);
    main0();
    m::fin();
}
