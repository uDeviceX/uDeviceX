#include <mpi.h>
#include <stdio.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "utils/mc.h"
#include "utils/imp.h"
#include "utils/error.h"
#include "conf/imp.h"
#include "io/mesh_read/imp.h"

#include "mesh/vectors/imp.h"
#include "mesh/vert_area/imp.h"

struct Out {
    MPI_Comm comm;
    MeshRead *mesh;
    const char *path;
};

static void dump(int nt, double *d, Vectors *pos, Out *out) {
    int id, nm;
    id = 0; nm = 1;
}

static void main0(const char *cell, Out *out) {
    int nv, nt, nm;
    MeshVertArea *vert_area;
    Vectors  *pos;
    double *vert_areas;
    UC(mesh_read_ini_off(cell, /**/ &out->mesh));
    UC(mesh_vert_area_ini(out->mesh, &vert_area));
    nv = mesh_read_get_nv(out->mesh);
    nt = mesh_read_get_nt(out->mesh);
    UC(vectors_float_ini(nv, mesh_read_get_vert(out->mesh), /**/ &pos));

    nm = 1;
    EMALLOC(nt, &vert_areas);
    mesh_vert_area_apply(vert_area, nm, pos, /**/ vert_areas);
    dump(nt, vert_areas, pos, out);

    mesh_vert_area_fin(vert_area);
    UC(vectors_fin(pos));
    UC(mesh_read_fin(out->mesh));
    EFREE(vert_areas);
}

int main(int argc, char **argv) {
    const char *i;
    Out out;
    Config *cfg;
    int rank, size, dims[3];
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &out.comm);

    MC(m::Comm_rank(out.comm, &rank));
    MC(m::Comm_size(out.comm, &size));

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(conf_lookup_string(cfg, "i", &i));
    UC(conf_lookup_string(cfg, "o", &out.path));

    main0(i, &out);

    UC(conf_fin(cfg));
    MC(m::Barrier(out.comm));
    m::fin();
}
