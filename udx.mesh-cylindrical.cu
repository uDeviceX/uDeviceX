// generated from udx.flat.cu
#include "udx.config.cuh"

/* begin src/mesh/cylindrical/imp.cpp */
#include <stdio.h>
#include <math.h>

/* begin src/utils/imp.h */
// tag::mem[]
void emalloc(size_t, /**/ void**); // <1>
void efree(void*); // <2>
void *ememcpy(void *dest, const void *src, size_t n); // <3>
// end::mem[]

// tag::macros[]
#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))
// end::macros[]

// tag::stdio[]
void efopen(const char *fname, const char *mode, /**/ FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);
// end::stdio[]

// tag::tools[]
bool same_str(const char *a, const char *b); // <1>
const char* get_filename_ext(const char*);
// end::tools[]
/* end src/utils/imp.h */
/* begin src/utils/error.h */
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)

/* udx check */
#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)

/* [B]o[p] [c]heck */
#define BPC(ans) do {                           \
        BopStatus s = (ans);                    \
        if (!bop_success(s)) {                  \
            ERR(":%s:%d: %s\n%s\n",             \
                __FILE__, __LINE__,             \
                bop_report_error_desc(s),       \
                bop_report_error_mesg());       \
        }} while (0)

void error_stack_push(const char *file, int line);
void error_stack_pop();

void error_signal(const char *file, int line, const char *fmt, ...);
void error_signal_cuda(const char *file, int line, const char *msg);
void error_signal_mpi(const char *file, int line, const char *msg);

bool error_get();
void error_report();
void error_abort();

void error_print_stack();
/* end src/utils/error.h */
/* begin src/utils/msg.h */
// tag::int[]
void msg_ini(int rank); // <1>
void msg_print(const char *fmt, ...); // <2>
// end::int[]
/* end src/utils/msg.h */

/* begin src/algo/vectors/imp.h */
struct Particle;
struct Vectors;
struct Coords;

// tag::interface[]
void vectors_float_ini(int n, const float*, /**/ Vectors**); // <1>
void vectors_postions_ini(int n, const Particle*, /**/ Vectors**); // <2>
void vectors_postions_edge_ini  (const Coords*, int n, const Particle*, /**/ Vectors**); // <3>
void vectors_postions_center_ini(const Coords*, int n, const Particle*, /**/ Vectors**); // <4>
void vectors_velocities_ini(int n, const Particle*, /**/ Vectors**); // <5>
void vectors_zero_ini(int n, /**/ Vectors**); // <6>

void vectors_fin(Vectors*);
void vectors_get(const Vectors*, int i, /**/ float r[3]); // <7>
// end::interface[]
/* end src/algo/vectors/imp.h */
/* begin src/algo/kahan_sum/imp.h */
struct KahanSum;

// tag::interface[]
void kahan_sum_ini(/**/ KahanSum**);
void kahan_sum_fin(KahanSum*);

void kahan_sum_add(KahanSum*, double input); // <1>
double kahan_sum_get(const KahanSum*); // <2>
double kahan_sum_compensation(const KahanSum*); // <3>
// end::interface[]
/* end src/algo/kahan_sum/imp.h */

/* begin src/mesh/cylindrical/imp.h */
struct MeshCylindrical;
struct Vectors;

// tag::interface[]
void mesh_cylindrical_ini(int nv, /**/ MeshCylindrical**);
void mesh_cylindrical_fin(MeshCylindrical*);
void  mesh_cylindrical_apply(MeshCylindrical*, int nm, Vectors*, /**/
                           double *r, double *phi, double *z);
// end::interface[]
/* end src/mesh/cylindrical/imp.h */
#include "udx.mesh-coordinate-common.cuh"
/* begin src/mesh/cylindrical/imp/type.h */
struct MeshCylindrical {
    double *rr; /* workspace for coordinates shifted to a mesh center
                   of mass */
    int nv;
};
/* end src/mesh/cylindrical/imp/type.h */
/* begin src/mesh/cylindrical/imp/main.h */
void mesh_cylindrical_ini(int nv, MeshCylindrical **pq) {
    MeshCylindrical *q;
    EMALLOC(1, &q);
    EMALLOC(3*nv, &q->rr);
    q->nv = nv;
    *pq = q;
}

void mesh_cylindrical_fin(MeshCylindrical *q) {
    EFREE(q->rr);
    EFREE(q);
}

static void xyz2cyl(double x, double y, double z, /**/ double *pr, double *pphi, double *pz) {
    double r, phi;
    r = sqrt(x*x + y*y);
    phi = atan2(y, x);
    *pr = r; *pphi  = phi; *pz = z;
}
static void mesh_cylindrical_apply_coordinates(int n, double *rr, /**/ double *r, double *phi, double *z) {
    enum {X, Y, Z};
    int i;
    for (i = 0; i < n; i++) {
        xyz2cyl(rr[X], rr[Y], rr[Z], /**/ r, phi, z);
        rr += 3; r++; phi++; z++;
    }
}
void mesh_cylindrical_apply(MeshCylindrical *q, int m, Vectors *pos, /**/ double *r, double *phi, double *z) {
    int i, nv, offset;
    double *rr;
    nv = q->nv; rr = q->rr; offset = 0;
    for (i = 0; i < m; i++) {
        UC(mesh_coordinate_to_com(nv, offset, pos, /**/ rr));
        mesh_cylindrical_apply_coordinates(nv, rr, /**/ r, phi, z);
        rr += 3*nv; r += nv; phi += nv; z += nv; offset += nv;
    }
}
/* end src/mesh/cylindrical/imp/main.h */
/* end src/mesh/cylindrical/imp.cpp */

#line 1 "src/mesh/dist/imp.cpp"
