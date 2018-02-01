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
#include "coords/imp.h"
#include "parser/imp.h"
#include "wall/wvel/type.h"

#include "wall/sdf/imp.h"
#include "math/tform/type.h"
#include "math/tform/dev.h"

#include "wall/sdf/tex3d/type.h"
#include "wall/sdf/type.h"

#include "wall/sdf/dev.h"
#include "wall/sdf/imp/type.h"

namespace dev {
#include "dev.h"
}

struct Part { float x, y, z; };

void main0(Sdf *sdf, Part *p) {
    Sdf_v sdf_v;
    float x, y, z;
    x = p->x; y = p->y; z = p->z;
    sdf_to_view(sdf, &sdf_v);
    KL(dev::main, (1, 1), (sdf_v, x, y, z));
}

void main1(const Coords *c, Part *p) {
    Sdf *sdf;
    int3 L;
    L = subdomain(c);
    UC(sdf_ini(L, &sdf));
    UC(sdf_gen(c, m::cart, true, sdf));
    UC(main0(sdf, p));
    UC(sdf_fin(sdf));
    dSync();    
}

void read_part(const Config *cfg, /**/ Part *p) {
    enum {X, Y, Z, D};
    float r[D];
    int n;

    UC(conf_lookup_vfloat(cfg, "pos", &n, r));
    
    p->x = r[X];
    p->y = r[Y];
    p->z = r[Z];
}

int main(int argc, char **argv) {
    Part p;
    Coords *coords;
    Config *cfg;    

    m::ini(&argc, &argv);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(read_part(cfg, /**/ &p));
    UC(coords_ini_conf(m::cart, cfg, &coords));

    UC(main1(coords, &p));

    UC(conf_fin(cfg));
    UC(coords_fin(coords));
    m::fin();
}
