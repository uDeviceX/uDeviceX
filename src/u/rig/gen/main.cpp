#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>

#include "inc/type.h"

#include "utils/mc.h"
#include "utils/msg.h"
#include "utils/error.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"

#include "conf/imp.h"
#include "coords/ini.h"
#include "coords/imp.h"

#include "struct/partlist/type.h"
#include "clist/imp.h"
#include "inter/color/imp.h"
#include "io/mesh_read/imp.h"
#include "flu/imp.h"
#include "rig/imp.h"
#include "rig/gen/imp.h"
#include "rigid/imp.h"

static void gen(MPI_Comm cart, const Config *cfg) {
    Coords *coords;
    GenColor *gc;
    FluQuants flu;
    RigQuants rig;
    MeshRead *mesh;
    int3 L;
    int maxp, maxs, numdensity;
    RigPinInfo *pi;
    RigGenInfo rgi;
    FluInfo fi;
    RigInfo ri;
    maxs = 200;

    UC(conf_lookup_int(cfg, "glb.numdensity", &numdensity));
    UC(coords_ini_conf(cart, cfg, &coords));
    L = subdomain(coords);
    maxp = 2 * numdensity * L.x * L.y * L.z;

    UC(flu_ini(false, false, L, maxp, &flu));
    UC(inter_color_ini(&gc));
    UC(inter_color_set_uniform(gc));
    UC(mesh_read_ini_ply("rig.ply", &mesh));
    UC(rig_ini(maxs, maxp, mesh, &rig));
    UC(flu_gen_quants(coords, numdensity, gc, &flu));

    UC(rig_gen_mesh(coords, cart, mesh, "rigs-ic.txt", /**/ &rig));

    UC(rig_pininfo_ini(&pi));
    UC(rig_pininfo_set_pdir(NOT_PERIODIC, pi));
    
    rgi.mass = 1.0;
    rgi.numdensity = numdensity;
    rgi.pi = pi;
    rgi.nt = mesh_read_get_nt(mesh);
    rgi.nv = mesh_read_get_nv(mesh);
    rgi.tt = rig.dtt;
    rgi.pp = rig.i_pp;
    rgi.empty_pp = false;    

    fi.pp = flu.pp;
    fi.n = &flu.n;

    ri.ns = rig.ns;
    ri.nps = &rig.nps;
    ri.n = &rig.n;
    ri.rr0 = rig.rr0_hst;
    ri.ss = rig.ss_hst;
    ri.pp = rig.pp_hst;
    
    UC(rig_gen_from_solvent(coords, cart, rgi, /*io*/ fi, /**/ ri));

    UC(rig_pininfo_fin(pi));
    UC(rig_fin(&rig));
    UC(mesh_read_fin(mesh));
    UC(inter_color_fin(gc));
    UC(flu_fin(&flu));    
    UC(coords_fin(coords));
}

int main(int argc, char **argv) {
    int rank, size, dims[3];
    MPI_Comm cart;
    Config *cfg;
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    MC(m::Comm_rank(cart, &rank));
    MC(m::Comm_size(cart, &size));
    msg_ini(rank);
    msg_print("mpi size: %d", size);

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));

    UC(gen(cart, cfg));
    
    UC(conf_fin(cfg));

    MC(m::Barrier(cart));
    m::fin();
}
