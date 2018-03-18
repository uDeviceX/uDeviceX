#include <mpi.h>
#include <stdio.h>

#include "utils/mc.h"
#include "utils/msg.h"
#include "mpi/glb.h"
#include "mpi/wrapper.h"
#include "utils/mc.h"
#include "utils/error.h"
#include "conf/imp.h"
#include "io/mesh_read/imp.h"
#include "rbc/matrices/imp.h"
#include "rbc/gen/imp.h"
#include "mesh/area/imp.h"
#include "mesh/volume/imp.h"
#include "mesh/vectors/imp.h"
#include "utils/imp.h"
#include "inc/type.h"

#define MAX_N 99999
#define MAX_M 20
static Particle pp[MAX_N];
static double output[MAX_M];

enum {AREA, VOLUME};
struct MeshQuant {
    int type;
    union {
        MeshArea *area;
        MeshVolume *volume;
    };
};

static void q_ini(const char *type, MeshRead *mesh, /**/ MeshQuant **pq) {
    MeshQuant *q;
    EMALLOC(1, &q);
    if (same_str(type, "area")) {
        q->type = AREA;
        mesh_area_ini(mesh, &q->area);
    } else if (same_str(type, "volume")) {
        q->type = VOLUME;
        mesh_volume_ini(mesh, &q->volume);
    } else
        ERR("unknown type '%s'", type);
    *pq = q;
}

static void q_apply(MeshQuant *q, int nm, Vectors *positions, double *out) {
    if (q->type == AREA)
        UC(mesh_area_apply(q->area, nm, positions, /**/ out));
    else if (q->type == VOLUME)
        UC(mesh_volume_apply(q->volume, nm, positions, /**/ out));
    else ERR("unknown q->type");
}

static void q_fin(MeshQuant *q) {
    if (q->type == AREA) mesh_area_fin(q->area);
    else if (q->type == VOLUME) mesh_volume_fin(q->volume);
    else ERR("unknown q->type");
    EFREE(q);
}

void main0(const char *cell, const char *ic, const char *quant) {
    int i, nm, n, nv;
    Matrices *matrices;
    MeshRead *mesh;
    MeshQuant *mesh_quant;
    Vectors *positions;
    const float *verts;
    UC(mesh_read_ini_off(cell, /**/ &mesh));
    UC(matrices_read(ic, &matrices));
    nv = mesh_read_get_nv(mesh);
    verts = mesh_read_get_vert(mesh);
    rbc_gen0(nv, verts, matrices, /**/ &n, pp);
    nm = n / nv;
    vectors_postions_ini(n, pp, /**/ &positions);
    UC(q_ini(quant, mesh, &mesh_quant));
    UC(q_apply(mesh_quant, nm, positions, /**/ output));
    for (i = 0; i < nm; i++)
        printf("%g\n", output[i]);
    
    UC(vectors_fin(positions));
    UC(q_fin(mesh_quant));
    UC(matrices_fin(matrices));
    UC(mesh_read_fin(mesh));
}

int main(int argc, char **argv) {
    const char *cell, *ic, *quant;
    Config *cfg;
    int rank, size, dims[3];
    MPI_Comm cart;
    m::ini(&argc, &argv);
    m::get_dims(&argc, &argv, dims);
    m::get_cart(MPI_COMM_WORLD, dims, &cart);

    MC(m::Comm_rank(cart, &rank));
    MC(m::Comm_size(cart, &size));

    UC(conf_ini(&cfg));
    UC(conf_read(argc, argv, cfg));
    UC(conf_lookup_string(cfg, "cell", &cell));
    UC(conf_lookup_string(cfg, "ic", &ic));
    UC(conf_lookup_string(cfg, "q", &quant));

    main0(cell, ic, quant);

    UC(conf_fin(cfg));
    MC(m::Barrier(cart));
    m::fin();
}
