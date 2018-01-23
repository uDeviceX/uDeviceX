#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#include <conf.h>
#include "inc/conf.h"

#include "d/ker.h"
#include "d/api.h"
#include "utils/msg.h"

#include "mpi/glb.h"
#include "mpi/wrapper.h"

#include "inc/type.h"
#include "inc/dev.h"
#include "utils/cc.h"
#include "utils/kl.h"
#include "utils/error.h" 
#include "coords/type.h"
#include "coords/ini.h"
#include "wall/wvel/type.h"

#include "wall/sdf/imp.h"
#include "math/tform/type.h"
#include "math/tform/dev.h"

#include "wall/sdf/tex3d/type.h"
#include "wall/sdf/type.h"

#include "wall/sdf/def.h"
#include "wall/sdf/dev.h"
#include "wall/sdf/imp/type.h"

namespace dev {
#include "dev.h"
}

struct Part { float x, y, z; };
static int    argc;
static char **argv;
/* left shift */
void lshift() {
    argc--;
    if (argc < 1) ERR("u/sdf: not enough args");
}

void main0(Sdf *sdf, Part *p) {
    Sdf_v sdf_v;
    float x, y, z;
    x = p->x; y = p->y; z = p->z;
    sdf_to_view(sdf, &sdf_v);
    KL(dev::main, (1, 1), (sdf_v, x, y, z));
}

void main1(Part *p) {
    Sdf *sdf;
    Coords *coords;
    UC(sdf_ini(&sdf));
    UC(coords_ini(m::cart, /**/ &coords));
    UC(sdf_gen(coords, m::cart, sdf));
    UC(main0(sdf, p));
    UC(sdf_fin(sdf));
    UC(coords_fin(coords));
    dSync();    
}

void ini_part(/**/ Part *p) {
    float x, y, z;
    x = atof(argv[argc - 1]); lshift();
    y = atof(argv[argc - 1]); lshift();
    z = atof(argv[argc - 1]); lshift();
    p->x = x; p->y = y; p->z = z;
}

void main2() {
    Part p;
    ini_part(/**/ &p);
    m::ini(&argc, &argv);
    UC(main1(&p));
    m::fin();
}

int main(int argc0, char **argv0) {
    argc = argc0; argv = argv0;
    UC(main2());
}
