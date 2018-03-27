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

#include "mesh/scalars/imp.h"
#include "mesh/angle/imp.h"
#include "mesh/vectors/imp.h"
#include "mesh/scatter/imp.h"

#include "mesh/tri_area/imp.h"
#include "io/vtk/imp.h"

struct Out {
    MPI_Comm comm;
    MeshRead *mesh;
    const char *path;
};

static void dump(double *tri_area,
                 Vectors *pos, Out *out) {
    enum {X, Y, Z};
    int nv, nt, nm, id;
    VTKConf *c;
    VTK *vtk;
    Scalars *area, *x, *y;
    nv = mesh_read_get_nv(out->mesh);
    nt = mesh_read_get_nt(out->mesh);
    nm = 1;

    UC(vtk_conf_ini(out->mesh, &c));
    UC(vtk_conf_tri(c, "area0"));
    UC(vtk_conf_tri(c, "area1"));
    UC(vtk_conf_vert(c, "x"));
    UC(vtk_conf_vert(c, "y"));
    vtk_ini(out->comm, nv, out->path, c, /**/ &vtk);
    vtk_points(vtk, nm, pos);

    scalars_double_ini(nt, tri_area, &area);
    scalars_vectors_ini(nv, pos, X, &x);
    scalars_vectors_ini(nv, pos, Y, &y);
    vtk_tri(vtk, nm, area, "area0");
    vtk_tri(vtk, nm, area, "area1");
    vtk_vert(vtk, nm, x, "x");
    vtk_vert(vtk, nm, y, "y");
    UC(scalars_fin(area));
    UC(scalars_fin(x));
    UC(scalars_fin(y));

    id = 0;
    UC(vtk_write(vtk, out->comm, id));

    UC(vtk_fin(vtk));
    UC(vtk_conf_fin(c));
}

static void compute_angle(MeshRead *mesh, int nm, Vectors *pos, double *angle_vert) {
    int ne;
    double *angle_edg;
    MeshAngle *angle;
    MeshScatter *scatter;
    Scalars *sc;

    ne = mesh_read_get_ne(mesh);
    EMALLOC(ne, &angle_edg);

    UC(mesh_angle_ini(mesh, &angle));
    UC(mesh_scatter_ini(mesh, &scatter));

    mesh_angle_apply(angle, nm, pos, /**/ angle_edg);
    UC(scalars_double_ini(ne, angle_edg, /**/ &sc));

    mesh_scatter_edg2vert(scatter, nm, sc, /**/ angle_vert);

    scalars_fin(sc);
    mesh_scatter_fin(scatter);
    mesh_angle_fin(angle);
    EFREE(angle_edg);
}
static void main0(const char *cell, Out *out) {
    int nv, nt, nm;
    MeshTriArea *tri_area;
    Vectors  *pos;
    double *tri_areas, *angle;
    UC(mesh_read_ini_off(cell, /**/ &out->mesh));
    UC(mesh_tri_area_ini(out->mesh, &tri_area));
    nv = mesh_read_get_nv(out->mesh);
    nt = mesh_read_get_nt(out->mesh);
    UC(vectors_float_ini(nv, mesh_read_get_vert(out->mesh), /**/ &pos));

    nm = 1;
    EMALLOC(nt, &tri_areas);
    EMALLOC(nv, &angle);
    compute_angle(out->mesh, nm, pos, /**/ angle);
    mesh_tri_area_apply(tri_area, nm, pos, /**/ tri_areas);
    dump(tri_areas, pos, out);

    mesh_tri_area_fin(tri_area);
    UC(vectors_fin(pos));
    UC(mesh_read_fin(out->mesh));
    EFREE(angle);
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
