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
#include "io/mesh/imp.h"

#include "mesh/vectors/imp.h"
#include "mesh/tri_area/imp.h"

#include "io/point/imp.h"

struct Out {
    MPI_Comm comm;
    MeshRead *mesh;
    const char *path;
};

static void dump(int nt, double *d,
                 Vectors *pos, Vectors *vel,
                 Out *out) {
    int id, nm;
    IOPointConf *c;
    IOPoint *p;
    MeshWrite *mesh_write;

    UC(io_point_conf_ini(&c));
    UC(io_point_conf_push(c, "area"));

    UC(io_point_ini(nt, out->path, c, &p));
    UC(mesh_write_ini_from_mesh(out->comm, out->mesh, "r", &mesh_write));

    UC(io_point_conf_fin(c));

    UC(io_point_push(p, nt, d, "area"));
    id = 0; nm = 1;
    UC(io_point_write(p, out->comm, id));
    UC(mesh_write_vectors(mesh_write, out->comm, nm, pos, vel, id));

    mesh_write_fin(mesh_write);
    UC(io_point_fin(p));
}

static void main0(const char *cell, Out *out) {
    int nv, nt, nm;
    MeshTriArea *tri_area;
    Vectors  *pos, *vel;
    double *tri_areas;
    UC(mesh_read_ini_off(cell, /**/ &out->mesh));
    UC(mesh_tri_area_ini(out->mesh, &tri_area));
    nv = mesh_read_get_nv(out->mesh);
    nt = mesh_read_get_nt(out->mesh);
    UC(vectors_float_ini(nv, mesh_read_get_vert(out->mesh), /**/ &pos));
    UC(vectors_zero_ini(nv,  /**/ &vel));

    nm = 1;
    EMALLOC(nt, &tri_areas);
    mesh_tri_area_apply(tri_area, nm, pos, /**/ tri_areas);
    dump(nt, tri_areas, pos, vel, out);

    mesh_tri_area_fin(tri_area);
    UC(vectors_fin(pos));
    UC(vectors_fin(vel));
    UC(mesh_read_fin(out->mesh));
    EFREE(tri_areas);
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
