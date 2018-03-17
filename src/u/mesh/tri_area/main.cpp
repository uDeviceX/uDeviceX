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

#include "mesh/positions/imp.h"
#include "mesh/tri_area/imp.h"

#include "io/point/imp.h"

#define PI (3.141592653589793)

struct Out {
    MPI_Comm comm;
    const char *path;
};

static void dump(int n, double *d, Out *out) {
    int id;
    IOPointConf *c;
    IOPoint *p;

    UC(io_point_conf_ini(&c));
    UC(io_point_conf_push(c, "area"));
    UC(io_point_ini(n, out->path, c, &p));
    UC(io_point_conf_fin(c));

    UC(io_point_push(p, n, d, "area"));
    id = 0;
    UC(io_point_write(p, out->comm, id));

    UC(io_point_fin(p));
}

static void main0(const char *cell, Out *out) {
    int i, nv, nt, nm;
    MeshRead *mesh;
    MeshTriArea *tri_area;
    Positions  *pos;
    double *tri_areas;
    UC(mesh_read_ini_off(cell, /**/ &mesh));
    UC(mesh_tri_area_ini(mesh, &tri_area));
    nv = mesh_read_get_nv(mesh);
    nt = mesh_read_get_nt(mesh);
    UC(positions_float_ini(nv, mesh_read_get_vert(mesh), /**/ &pos));

    nm = 1;
    EMALLOC(nt, &tri_areas);
    mesh_tri_area_apply(tri_area, nm, pos, /**/ tri_areas);
    for (i = 0; i < nt; i++)
        printf("%g\n", tri_areas[i]);
    dump(nt, tri_areas, out);
    
    mesh_tri_area_fin(tri_area);
    UC(positions_fin(pos));
    UC(mesh_read_fin(mesh));
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
