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
#include "mesh/tri_area/imp.h"
#include "io/vtk/imp.h"

struct Out {
    MPI_Comm comm;
    MeshRead *mesh;
    const char *path;
};

static void dump(double *tri_area,
                 Vectors *pos, Out *out) {
    int nv, nm, id;
    VTKConf *c;
    VTK *vtk;
    nv = mesh_read_get_nv(out->mesh);
    nm = 1;

    UC(vtk_conf_ini(out->mesh, &c));
    UC(vtk_conf_tri(c, "area"));
    vtk_ini(nv, out->path, c, /**/ &vtk);
    vtk_points(vtk, nm, pos);
    id = 0;
    vtk_write(vtk, out->comm, id);
    
    UC(vtk_fin(vtk));
    UC(vtk_conf_fin(c));
}

static void main0(const char *cell, Out *out) {
    int nv, nt, nm;
    MeshTriArea *tri_area;
    Vectors  *pos;
    double *tri_areas;
    UC(mesh_read_ini_off(cell, /**/ &out->mesh));
    UC(mesh_tri_area_ini(out->mesh, &tri_area));
    nv = mesh_read_get_nv(out->mesh);
    nt = mesh_read_get_nt(out->mesh);
    UC(vectors_float_ini(nv, mesh_read_get_vert(out->mesh), /**/ &pos));

    nm = 1;
    EMALLOC(nt, &tri_areas);
    mesh_tri_area_apply(tri_area, nm, pos, /**/ tri_areas);
    dump(tri_areas, pos, out);

    mesh_tri_area_fin(tri_area);
    UC(vectors_fin(pos));
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
