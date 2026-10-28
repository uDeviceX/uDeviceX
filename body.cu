#include "macros.cuh"










#include "config.cuh"

#include <mpi.h>
#include <string.h>
#include <vector_types.h>



#ifndef UDX_RIG_CORE_TYPE
#define UDX_RIG_CORE_TYPE
#ifndef UDX_RBC_RIG_TYPE_COMMON
#define UDX_RBC_RIG_TYPE_COMMON
#include "types.cuh"
#endif
#endif





#define D2D d::MemcpyDeviceToDevice
#define D2H d::MemcpyDeviceToHost
#define H2D d::MemcpyHostToDevice
#define H2H d::MemcpyHostToHost
#define A2A d::MemcpyDefault

#define cD2D(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), D2D))
#define cH2H(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), H2H))
#define cA2A(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), A2A))
#define cD2H(H, D, n) CC(d::Memcpy((H), (D), (n) * sizeof((H)[0]), D2H))
#define cH2D(D, H, n) CC(d::Memcpy((D), (H), (n) * sizeof((H)[0]), H2D))

#define aD2D(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), D2D))
#define aH2H(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), H2H))
#define aA2A(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), A2A))
#define aD2H(H, D, n) CC(d::MemcpyAsync((H), (D), (n) * sizeof((H)[0]), D2H))
#define aH2D(D, H, n) CC(d::MemcpyAsync((D), (H), (n) * sizeof((H)[0]), H2D))


#define Dfree(D)      CC(d::Free(D))
#define Dalloc(D, n)  CC(d::Malloc((void**)(D), (n) * sizeof(**(D))))


#define Dset(P, v, n) CC(d::Memset(P, v, (n)*sizeof(*(P))))
#define Dzero(P, n)   Dset(P, 0, n)


#define DsetA(P, v, n) CC(d::MemsetAsync(P, v, (n)*sizeof(*(P))))
#define DzeroA(P, n)   DsetA(P, 0, n)
#ifndef UDX_RIG_DEF_COMMON
#define UDX_RIG_DEF_COMMON
#ifndef UDX_RBC_RIG_DEF_COMMON
#define UDX_RBC_RIG_DEF_COMMON
#include "constants.cuh"
#endif
#endif

#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)


#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)


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
void msg_ini(int rank);
void msg_print(const char *fmt, ...);
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}


#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}





#ifndef UDX_RIG_DEVICE_API_COMMON
#define UDX_RIG_DEVICE_API_COMMON
#ifndef UDX_RBC_RIG_DEVICE_API_COMMON
#define UDX_RBC_RIG_DEVICE_API_COMMON
#include "device.cuh"
#endif
#endif
namespace m {
int Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int Barrier(MPI_Comm comm);
int Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
int Cancel(MPI_Request *request);
int Cart_rank(MPI_Comm comm, const int coords[], int *rank);
int Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart);
int Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]);
int Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]);
int Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
int Comm_free(MPI_Comm *comm);
int Comm_rank(MPI_Comm comm, int *rank);
int Comm_size( MPI_Comm comm, int *size );
int Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
int File_close(MPI_File *fh);
int File_get_position(MPI_File fh, MPI_Offset *offset);
int File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);
int File_seek(MPI_File fh, MPI_Offset offset, int whence);
int File_set_size(MPI_File fh, MPI_Offset size);
int File_write_at_all(MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
int Finalize(void);
int Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
           int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
int Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
            const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm);
int Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
              int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
int Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
               const int displs[], MPI_Datatype recvtype, MPI_Comm comm);
int Init(int *argc, char ***argv);
int Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
int Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
int Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
int Type_commit(MPI_Datatype *type);
int Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
int Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses);
int Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
int Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);
int Error_string(int errorcode, char *string, int *resultlen);


bool is_master(MPI_Comm comm);
int is_success(int errorcode);
int is_pending(int errorcode);
int is_err_in_status(int errorcode);
int status2errcode(MPI_Status *status);
}

void linal_inv3x3(const float m0[6],      float minv0[6]);
struct int4;
struct MeshRead;

void mesh_read_ini_off(const char *path, MeshRead**);
void mesh_read_ini_ply(const char *path, MeshRead**);
void mesh_read_fin(MeshRead*);

int mesh_read_get_nt(const MeshRead*);
int mesh_read_get_nv(const MeshRead*);
int mesh_read_get_ne(const MeshRead*);
int mesh_read_get_md(const MeshRead*);

const int4  *mesh_read_get_tri(const MeshRead*);
const float *mesh_read_get_vert(const MeshRead*);
const int4  *mesh_read_get_dih(MeshRead*);
struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl,      float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc,      float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl,      float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg,      float3 *rl);



int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);


bool is_end(const Coords *c, int dir);


void coord_stamp(const Coords *c,      char *s);


int coords_size(const Coords *c);
typedef unsigned char data_t;

enum {
    NFRAGS = 26,
    BULK   = 26,
    NBAGS  = 27,
};


enum AllocMod {
    HST_ONLY,
    DEV_ONLY,
    PINNED,
    PINNED_HST,
    PINNED_DEV,
    NONE
};

struct dBags {
    data_t *data[NBAGS];
};

struct hBags {
    data_t *data[NBAGS];
    int         *counts;
    int capacity[NBAGS];
    size_t bsize;
};

struct Comm;
struct CommBuffer;

void comm_bags_ini(AllocMod fmod, AllocMod bmod, size_t bsize, const int capacity[NBAGS],      hBags *hb, dBags *db);
void comm_bags_fin(AllocMod fmod, AllocMod bmod,      hBags *hb, dBags *db);
void comm_ini(MPI_Comm cart,      Comm **c);
void comm_fin(     Comm *c);
void comm_buffer_ini(int nbags, const hBags *hbb, CommBuffer**);
void comm_buffer_fin(CommBuffer*);


void comm_buffer_set(int nbags, const hBags*, CommBuffer*);
void comm_buffer_get(const CommBuffer*, int nbags, hBags*);

void comm_post_recv(hBags *b, Comm *c);
void comm_post_send(const hBags *b, Comm *c);

void comm_wait_recv(Comm *c,      hBags *b);
void comm_wait_send(Comm *c);

void comm_post_recv(CommBuffer *cb, Comm *c);
void comm_post_send(const CommBuffer *cb, Comm *c);

void comm_wait_recv(Comm *c,      CommBuffer *cb);
void comm_wait_send(Comm *c);

int    comm_get_number_capacity(int i, const hBags *b);
size_t comm_get_byte_capacity(int i, const hBags *b);

struct EMeshPack;
struct EMeshComm;
struct EMeshUnpack;
struct EMeshPackM;
struct EMeshCommM;
struct EMeshUnpackM;



void emesh_pack_ini(int3 L, int nv, int max_mesh_num, EMeshPack **p);
void emesh_comm_ini(MPI_Comm comm,      EMeshComm **c);
void emesh_unpack_ini(int3 L, int nv, int max_mesh_num, EMeshUnpack **u);

void emesh_pack_fin(EMeshPack *p);
void emesh_comm_fin(EMeshComm *c);
void emesh_unpack_fin(EMeshUnpack *u);

void emesh_build_map(int nm, int nv, const Particle *pp,      EMeshPack *p);

void emesh_pack_pp(int nv, const Particle *pp,      EMeshPack *p);
void emesh_pack_rrcp(int nv, const Positioncp *rr,      EMeshPack *p);
void emesh_download(EMeshPack *p);

void emesh_post_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_post_send(EMeshPack *p, EMeshComm *c);
void emesh_wait_recv(EMeshComm *c, EMeshUnpack *u);
void emesh_wait_send(EMeshComm *c);

void emesh_unpack_pp(int nv, const EMeshUnpack *u,      int *nmhalo, Particle *pp);
void emesh_unpack_rrcp(int nv, const EMeshUnpack *u,      int *nmhalo, Positioncp *rr);

void emesh_get_num_frag_mesh(const EMeshUnpack *u,      int cc[NFRAGS]);



void emesh_packm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshPackM **p);
void emesh_commm_ini(MPI_Comm comm,      EMeshCommM **c);
void emesh_unpackm_ini(int num_mom_per_mesh, int max_mesh_num, EMeshUnpackM **u);

void emesh_packm_fin(EMeshPackM *p);
void emesh_commm_fin(EMeshCommM *c);
void emesh_unpackm_fin(EMeshUnpackM *u);

void emesh_packM(int nt, const int counts[NFRAGS], const Momentum *mm,      EMeshPackM *p);
void emesh_downloadM(const int counts[NFRAGS], EMeshPackM *p);

void emesh_post_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_post_send(EMeshPackM *p, EMeshCommM *c);
void emesh_wait_recv(EMeshCommM *c, EMeshUnpackM *u);
void emesh_wait_send(EMeshCommM *c);

void emesh_upload(EMeshUnpackM *u);
void emesh_unpack_mom(int nt, const EMeshPack *p, const EMeshUnpackM *u,      Momentum *mm);
struct Particle;
struct float3;
void minmax(const Particle *pp, int nv, int nobj,      float3 *lo, float3 *hi);
struct Triangles {
    int4 *tt;
    int nt;
};
struct float3;
struct Particle;
struct int4;
struct Triangles;

int collision_inside_1p(int spdir, const float *r, const float *vv, const int4 *tt, const int nt);

void collision_label_ini(long n, int lab_out,      int *labels);
void collision_label(int pdir, int n, const Particle *pp, const Triangles *tri,
                     int nv, int nm, const Particle *i_pp,
                     const float3 *minext, const float3 *maxext,
                     int lab_in,      int *labels);
struct int4;
float mesh_volume0(int nt, const int4 *tt, const float *vv);
void mesh_center_of_mass(int nt, const int4 *tt, const float *vv,      float *com);
void mesh_inertia_tensor(int nt, const int4 *tt, const float *vv, const float *com, const float density,      float *I);
struct int4;
float dist_from_mesh(int nt, const int4 *tt, const float *vv, const float *r0);
enum {frag_bulk = 26};

struct int3;

namespace frag_hst {

int i2dx(int i);
int i2dy(int i);
int i2dz(int i);
void i2d3(int i,      int d[3]);

int d2i(int x, int y, int z);
int d32i(const int d[3]);

int ncell(int3 L, int i);

int antid2i(int x, int y, int z);
int anti(int i);

void estimates(int3 L, int nfrags, float maxd,      int *cap);

}

struct RigPinInfo;
struct int3;
struct Config;

#ifndef UDX_RIG_PERIODIC
#define UDX_RIG_PERIODIC
enum {NOT_PERIODIC = -1};
#endif

void rig_pininfo_ini(RigPinInfo **);
void rig_pininfo_fin(RigPinInfo *);
void rig_pininfo_set(int3 com, int3 axis, RigPinInfo *);
void rig_pininfo_set_pdir(int pdir, RigPinInfo *);
void rig_pininfo_set_conf(const Config *cfg, const char *ns, RigPinInfo *);

int rig_pininfo_get_pdir(const RigPinInfo *);

struct Particle;
struct Rigid;
struct Force;

void rig_reinit_ft(const int ns,      Rigid *ss);
void rig_update(const RigPinInfo *pi, float dt, int n, const Force *ff, const float *rr0, int ns,      Particle *pp, Rigid *ss);
void rig_generate(int ns, const Rigid *ss, int nps, const float *rr0,      Particle *pp);
void rig_update_mesh(float dt, int ns, const Rigid *ss, int nv, const float *vv,      Particle *pp);


#ifndef UDX_RIG_GEN_INFO
#define UDX_RIG_GEN_INFO
struct Coords;
struct RigPinInfo;
struct MeshRead;

struct RigGenInfo {
    float mass;
    int numdensity;
    const RigPinInfo *pi;
    int nt, nv;
    const int4 *tt;
    const Particle *pp;
    bool empty_pp;
    const MeshRead *mesh;
};

struct FluInfo {
    Particle *pp;
    int *n;
};


struct RigInfo {
    int ns, *nps;
    float *rr0;
    Rigid *ss;
    Particle *pp;
};

void rig_gen_from_solvent(const Coords*, MPI_Comm, RigGenInfo,        FluInfo,      RigInfo);
#endif

#define _I_ static
#define _S_ static

enum {
    IN  = -2,
    OUT = -1
};

_I_ int get_root(MPI_Comm comm, bool hasid0) {
    int rank, root0, root;
    MC(m::Comm_rank(comm, &rank));
    root0 = hasid0 ? rank : 0;
    root = 0;
    MC(m::Allreduce(&root0, &root, 1, MPI_INT, MPI_SUM, comm));
    if (hasid0 && root != rank) ERR("More than one rank has id 0");
    return root;
}

_I_ void exchange_mesh(int maxm, int3 L, MPI_Comm cart, int nv,        int *nm, Particle *pp,      int *cc) {
    EMeshPack *pack;
    EMeshComm *comm;
    EMeshUnpack *unpack;
    int nm0, nmhalo;
    nm0 = *nm;

    UC(emesh_pack_ini(L, nv, maxm, &pack));
    UC(emesh_comm_ini(cart,      &comm));
    UC(emesh_unpack_ini(L, nv, maxm, &unpack));

    UC(emesh_build_map(nm0, nv, pp,      pack));
    UC(emesh_pack_pp(nv, pp,      pack));
    UC(emesh_download(pack));

    UC(emesh_post_recv(comm, unpack));
    UC(emesh_post_send(pack, comm));
    UC(emesh_wait_recv(comm, unpack));
    UC(emesh_wait_send(comm));

    UC(emesh_unpack_pp(nv, unpack,      &nmhalo, pp + nm0 * nv));
    if (cc) UC(emesh_get_num_frag_mesh(unpack,      cc));
    *nm += nmhalo;

    UC(emesh_pack_fin(pack));
    UC(emesh_comm_fin(comm));
    UC(emesh_unpack_fin(unpack));
}

_I_ void compute_labels(int pdir, int n, const Particle *pp, int nt, int nv, int nm, const int4 *tt, const Particle *pp_mesh, int, int,      int *ll) {
    float3 *lo, *hi;
    Triangles tri;

    if (nm) Dalloc(&lo, nm);
    if (nm) Dalloc(&hi, nm);

    tri.nt = nt;
    tri.tt = (int4*) tt;

    if (nm) UC(minmax(pp_mesh, nv, nm,      lo, hi));
    UC(collision_label_ini(n, OUT,      ll));
    UC(collision_label(pdir, n, pp, &tri, nv, nm, pp_mesh, lo, hi, IN,      ll));

    if (nm) Dfree(lo);
    if (nm) Dfree(hi);
}
_S_ int3 fid2shift(int3 L, int fid) {
    int3 s;
    using namespace frag_hst;
    s.x = - i2dx(fid) * L.x;
    s.y = - i2dy(fid) * L.y;
    s.z = - i2dz(fid) * L.z;
    return s;
}

_S_ void extract_and_shift_hst(int3 s, int n, const Particle *pp, const int *labels, int *nx, float *rrx) {
    enum {X, Y, Z};
    int i, j;
    const float *r;
    float *rx;
    j = *nx;
    for (i = 0; i < n; ++i) {
        if (labels[i] == IN) {
            r = pp[i].r;
            rx = &rrx[3 * j];
            rx[X] = r[X] + s.x;
            rx[Y] = r[Y] + s.y;
            rx[Z] = r[Z] + s.z;
            ++j;
        }
    }
    *nx = j;
}

_S_ void label_extract_and_shift(int3 shift, int pdir, int n, const Particle *pp_dev, const Particle *pp_hst, int nt, int nv,
                                 int nm, const int4 *tt, const Particle *pp_mesh,
                                      int *ntempl, float *rrtempl,       int *ll_dev, int *ll_hst) {
    UC(compute_labels(pdir, n, pp_dev, nt, nv, nm, tt, pp_mesh, IN, OUT,      ll_dev));
    cD2H(ll_hst, ll_dev, n);
    extract_and_shift_hst(shift, n, pp_hst, ll_hst, ntempl, rrtempl);
}

_S_ void collect_and_broadcast_template(MPI_Comm comm, int *n, float *rr) {
    int i, rank, size, *starts, *counts, ntot;
    float *rr_recv;
    MC(m::Comm_rank(comm, &rank));
    MC(m::Comm_size(comm, &size));

    EMALLOC(size, &counts);
    EMALLOC(size, &starts);

    MC(m::Allgather(n, 1, MPI_INT, counts, 1, MPI_INT, comm));

    starts[0] = 0;
    for (i = 0; i < size; ++i) counts[i] *= 3;
    for (i = 1; i < size; ++i) starts[i] = starts[i-1] + counts[i-1];
    ntot = starts[size-1] + counts[size-1];

    EMALLOC(ntot, &rr_recv);

    MC(m::Allgatherv(rr, counts[rank], MPI_FLOAT, rr_recv, counts, starts, MPI_INT, comm));

    *n = ntot/3;
    memcpy(rr, rr_recv, ntot * sizeof(float));

    EFREE(rr_recv);
    EFREE(starts);
    EFREE(counts);
}

_S_ void label_template_dev(int pdir, int3 L, MPI_Comm cart, int nt, int nv, int nm, const int4 *tt, const Particle *pp_mesh,
                            int nflu, const Particle *pp_dev, const Particle *pp_hst,      int *nps, float *rr0,       int *ll_dev, int *ll_hst) {
    int i, maxm, nmall, n, cc[NFRAGS];
    int3 shift;
    Particle *pp0, *pp;

    maxm = NFRAGS + 1;
    Dalloc(&pp, nv * maxm);
    pp0 = pp;

    nmall = nm;
    n = nm * nv;
    if (n) cD2D(pp, pp_mesh, n);

    UC(exchange_mesh(maxm, L, cart, nv,          &nmall, pp,      cc));


    shift = fid2shift(L, frag_bulk);
    if (nm) UC(label_extract_and_shift(shift, pdir, nflu, pp_dev, pp_hst, nt, nv, nm, tt, pp,      nps, rr0,       ll_dev, ll_hst));
    pp += nm * nv;


    for (i = 0; i < NFRAGS; ++i) {
        nm = cc[i];
        shift = fid2shift(L, i);
        if (nm) UC(label_extract_and_shift(shift, pdir, nflu, pp_dev, pp_hst, nt, nv, nm, tt, pp,      nps, rr0,       ll_dev, ll_hst));
        pp += nm * nv;
    }

    UC(collect_and_broadcast_template(cart, nps, rr0));

    Dfree(pp0);
}

struct Transf {
    float s[3];
    float e0[3], e1[3], e2[3];
};

_S_ void get_transf(MPI_Comm comm, bool hasid0, const Rigid *ss, Transf *T) {
    int root, sz;
    if (hasid0) {
        sz = 3 * sizeof(float);
        memcpy(T->s, ss[0].com, sz);
        memcpy(T->e0, ss[0].e0, sz);
        memcpy(T->e1, ss[0].e1, sz);
        memcpy(T->e2, ss[0].e2, sz);
    }
    root = get_root(comm, hasid0);
    MC(m::Bcast(T->s, 3, MPI_FLOAT, root, comm));
    MC(m::Bcast(T->e0, 3, MPI_FLOAT, root, comm));
    MC(m::Bcast(T->e1, 3, MPI_FLOAT, root, comm));
    MC(m::Bcast(T->e2, 3, MPI_FLOAT, root, comm));
}

_S_ float dot(const float a[], const float b[]) {
    enum {X, Y, Z};
    return a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z];
}

_S_ void transform(const Transf *T, float *r) {
    enum {X, Y, Z};
    int c;
    float r0[3];
    for (c = 0; c < 3; ++c) r0[c] = r[c] - T->s[c];
    r[X] = dot(r0, T->e0);
    r[Y] = dot(r0, T->e1);
    r[Z] = dot(r0, T->e2);
}

_S_ void transf_template(const Transf *T, int n, float *rr) {
    int i;
    for (i = 0; i < n; ++i) transform(T, &rr[3*i]);
}

_I_ void extract_template(int3 L, MPI_Comm cart, RigGenInfo rgi, int n, const Particle *flu_pp_dev, const Particle *flu_pp_hst,
                             int, bool hasid0, const Rigid *ss,      int *nps, float *rr0,       int *ll_dev, int *ll_hst) {
    int nm, pdir;
    Transf T;
    *nps = 0;
    nm = hasid0 ? 1 : 0;
    pdir = rig_pininfo_get_pdir(rgi.pi);

    UC(label_template_dev(pdir, L, cart, rgi.nt, rgi.nv, nm, rgi.tt, rgi.pp, n, flu_pp_dev, flu_pp_hst,      nps, rr0,       ll_dev, ll_hst));
    UC(get_transf(cart, hasid0, ss, &T));
    UC(transf_template(&T, *nps, rr0));
}

_I_ void empty_solid(const MeshRead *mesh,          int *n, float *rr) {
    enum {X, Y, Z};
    int n0, i, j, nt;
    const float *r, *vv;
    const int4 *tt;
    float d;
    n0 = *n;
    nt = mesh_read_get_nt(mesh);
    tt = mesh_read_get_tri(mesh);
    vv = mesh_read_get_vert(mesh);

    for (i = j = 0; i < n0; ++i) {
        r = rr + 3*i;
        d = dist_from_mesh(nt, tt, vv, r);
        if (d <= 1.0) {
            rr[3*j + X] = r[X];
            rr[3*j + Y] = r[Y];
            rr[3*j + Z] = r[Z];
            ++j;
        }
    }
    if (j == 0)
        ERR("No particle remaining in solid template (size = %d)", n0);

    msg_print("Template solid: keep %d out of %d particles", j, n0);
    *n = j;
}
_S_ void remove_hst(const int *labels, int *n, Particle *pp) {
    int i, j;
    for (i = j = 0; i < *n; ++i) {
        if (labels[i] == OUT)
            pp[j++] = pp[i];
    }
    *n = j;
}

_I_ void kill_solvent(RigGenInfo rgi, int maxm, int3 L, MPI_Comm cart, int nm,
                                int *n, Particle *pp_hst, Particle *pp_dev,
                               int *ll_dev, int *ll_hst) {
    int nmall, pdir;
    Particle *i_ppall;
    nmall = nm;
    pdir = rig_pininfo_get_pdir(rgi.pi);

    Dalloc(&i_ppall, maxm * rgi.nv);
    cD2D(i_ppall, rgi.pp, nm * rgi.nv);

    UC(exchange_mesh(maxm, L, cart, rgi.nv,        &nmall, i_ppall, NULL));
    UC(compute_labels(pdir, *n, pp_dev, rgi.nt, rgi.nv, nmall, rgi.tt, i_ppall, IN, OUT,      ll_dev));

    cD2H(ll_hst, ll_dev, *n);
    remove_hst(ll_hst, n, pp_hst);
    cH2D(pp_dev, pp_hst, *n);

    Dfree(i_ppall);
}
_S_ void init_I_from_pos(int n, const float *rr, float pmass,      float *I) {
    enum {X, Y, Z};
    enum {XX, XY, XZ, YY, YZ, ZZ};
    enum {YX = XY, ZX = XZ, ZY = YZ};
    int c, i;
    const float *r;
    float x, y, z;

    for (c = 0; c < 6; ++c) I[c] = 0;
    for (i = 0; i < n; ++i) {
        r = &rr[3*i];
        x = r[X]; y = r[Y]; z = r[Z];
        I[XX] += y*y + z*z;
        I[YY] += z*z + x*x;
        I[ZZ] += x*x + y*y;
        I[XY] -= x*y;
        I[XZ] -= z*x;
        I[YZ] -= y*z;
    }
    for (c = 0; c < 6; ++c) I[c] *= pmass;
}

_S_ void init_I_from_mesh(float density, int nt, const int4 *tt, const float *vv,      float *I) {
    float com[3] = {0};
    mesh_center_of_mass(nt, tt, vv,      com);
    mesh_inertia_tensor(nt, tt, vv, com, density,      I);
}

_S_ void compute_properties(const RigPinInfo *pi, int n, const float *rr0, float pmass,
                               float numdensity, const MeshRead *mesh,      Rigid *s) {
    enum {X, Y, Z};
    int spdir, nt;
    const int4 *tt;
    const float *vv;
    float I[6], rho;

    spdir = rig_pininfo_get_pdir(pi);



    if (spdir == NOT_PERIODIC) {
        rho = pmass * numdensity;
        nt = mesh_read_get_nt(mesh);
        tt = mesh_read_get_tri(mesh);
        vv = mesh_read_get_vert(mesh);
        init_I_from_mesh(rho, nt, tt, vv,      I);
        s->mass = rho * mesh_volume0(nt, tt, vv);
    }
    else {
        init_I_from_pos(n, rr0, pmass,      I);
        s->mass = n * pmass;
    }

    UC(linal_inv3x3(I,      s->Iinv));
}

_S_ void copy_props(const Rigid *s0, Rigid *s) {
    s->mass = s0->mass;
    memcpy(s->Iinv, s0->Iinv, 6*sizeof(float));
}

_S_ void clear_vel(Rigid *s) {
    enum {X, Y, Z};
    s->v[X] = s->v[Y] = s->v[Z] = 0;
    s->om[X] = s->om[Y] = s->om[Z] = 0;
}

_S_ void clear_forces(Rigid *s) {
    enum {X, Y, Z};
    s->fo[X] = s->fo[Y] = s->fo[Z] = 0;
    s->to[X] = s->to[Y] = s->to[Z] = 0;
}

_I_ void set_properties(MPI_Comm, RigGenInfo rgi, int n, const float *rr0, int ns, const int *ids,      Rigid *ss) {
    Rigid s_props, *s;
    int i;

    compute_properties(rgi.pi, n, rr0, rgi.mass, rgi.numdensity, rgi.mesh, &s_props);

    for (i = 0; i < ns; ++i) {
        s = &ss[i];
        copy_props(&s_props, s);
        clear_vel(s);
        clear_forces(s);
        s->id = ids[i];
    }
}
static void gen_ids(MPI_Comm comm, int n, int *ii) {
    int i, i0 = 0, count = 1;
    MC(m::Exscan(&n, &i0, count, MPI_INT, MPI_SUM, comm));
    for (i = 0; i < n; ++i) ii[i] = i + i0;
}

void rig_gen_from_solvent(const Coords *coords, MPI_Comm cart, RigGenInfo rgi,        FluInfo fi,      RigInfo ri) {
    int3 L;
    int *ids, nflu, *ll_hst, *ll_dev;
    Particle *pp_flu_hst;
    bool hasid0;

    L = subdomain(coords);
    nflu = *fi.n;

    EMALLOC(ri.ns, &ids);
    EMALLOC(nflu, &pp_flu_hst);
    EMALLOC(nflu, &ll_hst);
    Dalloc(&ll_dev, nflu);
    cD2H(pp_flu_hst, fi.pp, nflu);

    UC(gen_ids(cart, ri.ns, ids));

    hasid0 = (ri.ns && ids[0] == 0);

    UC(extract_template(L, cart, rgi, nflu, fi.pp, pp_flu_hst, ri.ns, hasid0, ri.ss,
                             ri.nps, ri.rr0,       ll_dev, ll_hst));

    UC(kill_solvent(rgi, MAX_SOLIDS, L, cart, ri.ns,
                           fi.n, pp_flu_hst, fi.pp,
                          ll_dev, ll_hst));

    UC(set_properties(cart, rgi, *ri.nps, ri.rr0, ri.ns, ids,      ri.ss));

    if (rgi.empty_pp)
        UC(empty_solid(rgi.mesh,        ri.nps, ri.rr0));
    Dfree(ll_dev);
    EFREE(pp_flu_hst);
    EFREE(ll_hst);
    EFREE(ids);
}

#undef _I_
#undef _S_





#include <mpi.h>
#include <stdio.h>
#include <vector_types.h>
#include <assert.h>
#include <string.h>
#include <math.h>




void msg_ini(int rank);
void msg_print(const char *fmt, ...);
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)


#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)


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

struct Coords;
struct Coords_v;
struct float3;
struct int3;

void coords_get_view(const Coords *c, Coords_v *v);

int xdomain(const Coords *c);
int ydomain(const Coords *c);
int zdomain(const Coords *c);

float xl2xc(const Coords *c, float xl);
float yl2yc(const Coords *c, float yl);
float zl2zc(const Coords *c, float zl);
void local2center(const Coords *c, float3 rl,      float3 *rc);

float xc2xl(const Coords *c, float xc);
float yc2yl(const Coords *c, float yc);
float zc2zl(const Coords *c, float zc);
void center2local(const Coords *c, float3 rc,      float3 *rl);

float xl2xg(const Coords *c, float xl);
float yl2yg(const Coords *c, float yl);
float zl2zg(const Coords *c, float zl);
void local2global(const Coords *c, float3 rl,      float3 *rg);

float xg2xl(const Coords *c, float xg);
float yg2yl(const Coords *c, float yg);
float zg2zl(const Coords *c, float zg);
void global2local(const Coords *c, float3 rg,      float3 *rl);



int xlo(const Coords*);
int ylo(const Coords*);
int zlo(const Coords*);

int xhi(const Coords*);
int yhi(const Coords*);
int zhi(const Coords*);

int xs(const Coords*);
int ys(const Coords*);
int zs(const Coords*);
int3 subdomain(const Coords*);


bool is_end(const Coords *c, int dir);


void coord_stamp(const Coords *c,      char *s);


int coords_size(const Coords *c);

#ifndef UDX_RIG_DEVICE_API_COMMON
#define UDX_RIG_DEVICE_API_COMMON
#include "device.cuh"
#endif





#define D2D d::MemcpyDeviceToDevice
#define D2H d::MemcpyDeviceToHost
#define H2D d::MemcpyHostToDevice
#define H2H d::MemcpyHostToHost
#define A2A d::MemcpyDefault

#define cD2D(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), D2D))
#define cH2H(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), H2H))
#define cA2A(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), A2A))
#define cD2H(H, D, n) CC(d::Memcpy((H), (D), (n) * sizeof((H)[0]), D2H))
#define cH2D(D, H, n) CC(d::Memcpy((D), (H), (n) * sizeof((H)[0]), H2D))

#define aD2D(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), D2D))
#define aH2H(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), H2H))
#define aA2A(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), A2A))
#define aD2H(H, D, n) CC(d::MemcpyAsync((H), (D), (n) * sizeof((H)[0]), D2H))
#define aH2D(D, H, n) CC(d::MemcpyAsync((D), (H), (n) * sizeof((H)[0]), H2D))


#define Dfree(D)      CC(d::Free(D))
#define Dalloc(D, n)  CC(d::Malloc((void**)(D), (n) * sizeof(**(D))))


#define Dset(P, v, n) CC(d::Memset(P, v, (n)*sizeof(*(P))))
#define Dzero(P, n)   Dset(P, 0, n)


#define DsetA(P, v, n) CC(d::MemsetAsync(P, v, (n)*sizeof(*(P))))
#define DzeroA(P, n)   DsetA(P, 0, n)
#ifndef UDX_RIG_DEF_COMMON
#define UDX_RIG_DEF_COMMON
#include "constants.cuh"
#endif
#ifndef UDX_RIG_CORE_TYPE
#define UDX_RIG_CORE_TYPE
#include "types.cuh"
#endif


#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}






struct int4;
struct MeshRead;

void mesh_read_ini_off(const char *path, MeshRead**);
void mesh_read_ini_ply(const char *path, MeshRead**);
void mesh_read_fin(MeshRead*);

int mesh_read_get_nt(const MeshRead*);
int mesh_read_get_nv(const MeshRead*);
int mesh_read_get_ne(const MeshRead*);
int mesh_read_get_md(const MeshRead*);

const int4  *mesh_read_get_tri(const MeshRead*);
const float *mesh_read_get_vert(const MeshRead*);
const int4  *mesh_read_get_dih(MeshRead*);
#ifndef UDX_RBC_RIG_RESTART_COMMON
#define UDX_RBC_RIG_RESTART_COMMON
struct Coords;
struct Particle;
struct Rigid;

enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};

typedef void (*StreamWriter)(const void*, FILE*);
typedef void (*StreamReader)(FILE*, void*);

void restart_write_pp(MPI_Comm, const char *base, const char *code, int id, long n, const Particle *pp);
void restart_write_ii(MPI_Comm, const char *base, const char *code, int id, long n, const int *ii);
void restart_write_ss(MPI_Comm, const char *base, const char *code, int id, long n, const Rigid *ss);
void restart_write_stream_one_node(MPI_Comm, const char *base, const char *code, int id, const void *data, StreamWriter);

void restart_read_pp(MPI_Comm, const char *base, const char *code, int id, int *n, Particle *pp);
void restart_read_ii(MPI_Comm, const char *base, const char *code, int id, int *n, int *ii);
void restart_read_ss(MPI_Comm, const char *base, const char *code, int id, int *n, Rigid *ss);
void restart_read_stream_one_node(const char *base, const char *code, int id, StreamReader, void *data);
#endif

struct Matrices;
struct Coords;

void matrices_read(const char *path,      Matrices**);
void matrices_read_filter(const char *path, const Coords*,      Matrices**);

void matrices_get(const Matrices*, int i,      double**);
void matrices_get_r(const Matrices*, int i,      double r[3]);
int  matrices_get_n(const Matrices*);
void matrices_log(const Matrices*);

void matrices_fin(Matrices*);
struct Coords;
struct Particle;
struct Matrices;


int  mesh_gen_from_file(const Coords*, const float *vv, const char *ic, int nv,      Particle*);
void mesh_gen_from_matrices(int nv, const float *vv, const Matrices*,      int *pn, Particle*);
void mesh_shift(const Coords*, int n,      Particle*);

struct RigQuants {
    int n, ns, nps;
    Particle *pp_hst, *pp;
    Rigid    *ss_hst, *ss;
    float   *rr0_hst, *rr0;


    int nt, nv;
    int4 *dtt;
    float *dvv;
    Particle *i_pp_hst, *i_pp;

    Rigid *ss_dmp;

    long maxp;
};

struct Coords;
struct RigPinInfo;
struct MeshRead;

void rig_ini(long maxs, long maxp, const MeshRead*, RigQuants*);
void rig_fin(RigQuants *q);

void rig_gen_mesh(const Coords*, MPI_Comm, const MeshRead*, const char *ic,      RigQuants *);
void rig_gen_freeze(const Coords*, bool empty_pp, int numdensity, float rig_mass, const RigPinInfo*,
                    MPI_Comm, const MeshRead*, Particle *opp, int *on, RigQuants *q);
void rig_strt_quants(MPI_Comm, const MeshRead*, const char *base, const char *name, const int id, RigQuants *q);

void rig_strt_dump_templ(MPI_Comm, const char *base, const char *name, const RigQuants *q);
void rig_strt_dump(MPI_Comm, const char *base, const char *name, const int id, const RigQuants *q);

#ifndef UDX_RIG_GEN_INFO
#define UDX_RIG_GEN_INFO
struct Coords;
struct RigPinInfo;
struct MeshRead;

struct RigGenInfo {
    float mass;
    int numdensity;
    const RigPinInfo *pi;
    int nt, nv;
    const int4 *tt;
    const Particle *pp;
    bool empty_pp;
    const MeshRead *mesh;
};

struct FluInfo {
    Particle *pp;
    int *n;
};


struct RigInfo {
    int ns, *nps;
    float *rr0;
    Rigid *ss;
    Particle *pp;
};

void rig_gen_from_solvent(const Coords*, MPI_Comm, RigGenInfo,        FluInfo,      RigInfo);
#endif

void rig_ini(long maxs, long maxp, const MeshRead *mesh, RigQuants *q) {
    q->n = q->ns = q->nps = 0;
    q->maxp = maxp;

    Dalloc(&q->pp,      maxp);
    Dalloc(&q->ss,      maxs);
    Dalloc(&q->rr0, 3 * maxp);
    Dalloc(&q->i_pp,    maxp);

    EMALLOC(maxp, &q->pp_hst);
    EMALLOC(maxs, &q->ss_hst);
    EMALLOC(3 * maxp, &q->rr0_hst);
    EMALLOC(maxp, &q->i_pp_hst);

    EMALLOC(maxs, &q->ss_dmp);

    q->nt = mesh_read_get_nt(mesh);
    q->nv = mesh_read_get_nv(mesh);

    Dalloc(&q->dtt,     q->nt);
    Dalloc(&q->dvv, 3 * q->nv);

    cH2D(q->dtt, mesh_read_get_tri (mesh),     q->nt);
    cH2D(q->dvv, mesh_read_get_vert(mesh), 3 * q->nv);
}
void rig_fin(RigQuants *q) {
    EFREE(q->pp_hst);
    EFREE(q->ss_hst);
    EFREE(q->rr0_hst);
    EFREE(q->i_pp_hst);

    Dfree(q->pp);
    Dfree(q->ss);
    Dfree(q->rr0);
    Dfree(q->i_pp);

    Dfree(q->dtt);
    Dfree(q->dvv);

    EFREE(q->ss_dmp);
}
enum {X, Y, Z};

static void update_r_hst(const float *rr0, const int n, const float *com, const float *e0, const float *e1, const float *e2,      Particle *pp) {
    for (int ip = 0; ip < n; ++ip) {
        float *r0 = pp[ip].r;
        const float* ro = &rr0[3*ip];
        float x = ro[X], y = ro[Y], z = ro[Z];
        r0[X] = x*e0[X] + y*e1[X] + z*e2[X];
        r0[Y] = x*e0[Y] + y*e1[Y] + z*e2[Y];
        r0[Z] = x*e0[Z] + y*e1[Z] + z*e2[Z];

        r0[X] += com[X]; r0[Y] += com[Y]; r0[Z] += com[Z];
    }
}

static void reinit_ft_hst(const int nsolid,      Rigid *ss) {
    for (int i = 0; i < nsolid; ++i) {
        Rigid *s = ss + i;

        s->fo[X] = s->fo[Y] = s->fo[Z] = 0;
        s->to[X] = s->to[Y] = s->to[Z] = 0;
    }
}

static void generate_hst(const Rigid *ss_hst, const int ns, const float *rr0, const int nps,      Particle *pp) {
    int j, start;
    for (j = start = 0; j < ns; ++j) {
        update_r_hst(rr0, nps, ss_hst[j].com, ss_hst[j].e0, ss_hst[j].e1, ss_hst[j].e2,      pp + start);
        start += nps;
    }
}

static void mesh2pp_hst(const Rigid *ss_hst, const int ns, int nv, const float *vv,      Particle *pp) {
    int i, j;
    for (j = 0; j < ns; ++j) {
        const Rigid *s = ss_hst + j;
        update_r_hst(vv, nv, s->com, s->e0, s->e1, s->e2,      pp + j * nv);

        for (i = 0; i < nv; ++i) {
            float *v = pp[j*nv + i].v;
            v[X] = v[Y] = v[Z] = 0;
        }
    }
}

static void gen_pp_hst(const int ns, const float *rr0_hst, const int nps,      Rigid *ss_hst, Particle *pp_hst) {
    generate_hst(ss_hst, ns, rr0_hst, nps,      pp_hst);
    reinit_ft_hst(ns,      ss_hst);
}

static void gen_ipp_hst(const Rigid *ss_hst, const int ns, int nv, const float *vv, Particle *i_pp_hst) {
    mesh2pp_hst(ss_hst, ns, nv, vv,      i_pp_hst);
}

static void cpy_H2D(const RigQuants *q) {
    cH2D(q->i_pp, q->i_pp_hst, q->ns * q->nv);
    cH2D(q->ss,   q->ss_hst,   q->ns);
    cH2D(q->rr0,  q->rr0_hst,  q->nps * 3);
    cH2D(q->pp,   q->pp_hst,   q->n);
}
static void check(const float e[]) {
    enum {X, Y, Z};
    float ee, err;
    ee = e[X]*e[X] + e[Y]*e[Y] + e[Z]*e[Z];
    err = fabs(ee - 1.f);
    if (err > 1e-3)
        ERR("rigid ic: Matrix should be rotation only. Do not support scaling\n"
            "error: %g, e = %g %g %g", err, e[X], e[Y], e[Z]);
}

static Rigid gen_from_matrix(const double *A) {
    enum {X, Y, Z, W, M};
    enum {D = W};
    int c;
    Rigid s; memset(&s, 0, sizeof(s));
    for (c = 0; c < D; ++c) {
        s.com[c] = A[M * c + W];
        s.e0[c]  = A[M * c + X];
        s.e1[c]  = A[M * c + Y];
        s.e2[c]  = A[M * c + Z];
    }
    UC(check(s.e0));
    UC(check(s.e1));
    UC(check(s.e2));
    return s;
}

static void gen_from_matrices(Matrices *matrices, Rigid *ss) {
    int i, n;
    double *A;
    n = matrices_get_n(matrices);

    for (i = 0; i < n; ++i) {
        matrices_get(matrices, i, &A);
        ss[i] = gen_from_matrix(A);
    }
}

static void shift(const Coords *c, int n, Rigid *ss) {
    enum {X, Y, Z};
    float *r;
    int i;
    for (i = 0; i < n; ++i) {
        r = ss[i].com;
        r[X] = xg2xl(c, r[X]);
        r[Y] = yg2yl(c, r[Y]);
        r[Z] = zg2zl(c, r[Z]);
    }
}

void rig_gen_mesh(const Coords *coords, MPI_Comm comm, const MeshRead *mesh, const char *ic,      RigQuants *q) {
    const float *vv;
    int n, nm, nv;
    Matrices *matrices;
    nv = mesh_read_get_nv(mesh);
    vv = mesh_read_get_vert(mesh);

    UC(matrices_read_filter(ic, coords,      &matrices));

    q->ns = nm = matrices_get_n(matrices);
    if (nm > MAX_SOLIDS)
        ERR("nm=%d >= MAX_SOLIDS=%d", nm, MAX_SOLIDS);

    UC(mesh_gen_from_matrices(nv, vv, matrices,      &n, q->i_pp_hst));
    UC(mesh_shift(coords, n, q->i_pp_hst));

    UC(gen_from_matrices(matrices, q->ss_hst));
    UC(shift(coords, nm, q->ss_hst));

    if (n)  cH2D(q->i_pp, q->i_pp_hst, n);
    if (nm) cH2D(q->ss, q->ss_hst, nm);

    UC(matrices_fin(matrices));
}

void rig_gen_freeze(const Coords *coords, bool empty_pp, int numdensity, float rig_mass, const RigPinInfo *pi, MPI_Comm comm,
                    const MeshRead *mesh,          Particle *opp, int *on,      RigQuants *q) {
    RigGenInfo rgi;
    FluInfo fluinfo;
    RigInfo riginfo;

    rgi.mass = rig_mass;
    rgi.numdensity = numdensity;
    rgi.pi = pi;
    rgi.nt = q->nt;
    rgi.nv = q->nv;
    rgi.tt = q->dtt;
    rgi.pp = q->i_pp;
    rgi.empty_pp = empty_pp;
    rgi.mesh = mesh;

    fluinfo.pp = opp;
    fluinfo.n = on;

    riginfo.ns = q->ns;
    riginfo.nps = &q->nps;
    riginfo.rr0 = q->rr0_hst;
    riginfo.ss = q->ss_hst;
    riginfo.pp = q->pp_hst;

    UC(rig_gen_from_solvent(coords, comm, rgi,        fluinfo,      riginfo));
    gen_pp_hst(q->ns, q->rr0_hst, q->nps,      q->ss_hst, q->pp_hst);
    gen_ipp_hst(q->ss_hst, q->ns, q->nv, mesh_read_get_vert(mesh),      q->i_pp_hst);
    cpy_H2D(q);
}
#define PP ".pp"
#define SS ".ss"

static void gen_code(const char *name, const char *ext, char *code) {
    strcpy(code, name);
    strcat(code, ext);
}

static void pp2rr(const Particle *pp, const int n, float *rr) {
    int i, c;
    for (i = 0; i < n; ++i)
        for ( c = 0; c < 3; ++c)
            rr[3*i + c] = pp[i].r[c];
}

static void gen_from_strt(int maxp, MPI_Comm comm, const char *base, const char *name, const int id, int *ns, int *nps, int *n, float *rr0_hst, Rigid *ss_hst) {
    Particle *pp;
    char code[FILENAME_MAX];

    EMALLOC(maxp, &pp);
    gen_code(name, PP, code);
    restart_read_pp(comm, base, code, RESTART_TEMPL, nps, pp);
    pp2rr(pp, *nps, rr0_hst);
    EFREE(pp);

    gen_code(name, SS, code);
    restart_read_ss(comm, base, code, id, ns, ss_hst);
    *n = *ns * (*nps);
}

void rig_strt_quants(MPI_Comm comm, const MeshRead *mesh, const char *base, const char *name, const int id, RigQuants *q) {
    gen_from_strt(q->maxp, comm, base, name, id,      &q->ns, &q->nps, &q->n, q->rr0_hst, q->ss_hst);
    gen_pp_hst(q->ns, q->rr0_hst, q->nps,      q->ss_hst, q->pp_hst);
    gen_ipp_hst(q->ss_hst, q->ns, q->nv, mesh_read_get_vert(mesh),      q->i_pp_hst);
    cpy_H2D(q);
}

static void rr2pp(const float *rr, const int n, Particle *pp) {
    for (int i = 0; i < n; ++i)
    for (int c = 0; c < 3; ++c) {
        pp[i].r[c] = rr[3*i + c];
        pp[i].v[c] = 0;
    }
}

static void strt_dump_templ0(MPI_Comm comm, const char *base, const char *name, const int nps, const float *rr0_hst) {
    Particle *pp;
    char code[FILENAME_MAX];
    EMALLOC(nps, &pp);
    rr2pp(rr0_hst, nps, pp);

    gen_code(name, PP, code);
    restart_write_pp(comm, base, code, RESTART_TEMPL, nps, pp);

    EFREE(pp);
}

void rig_strt_dump_templ(MPI_Comm comm, const char *base, const char *name, const RigQuants *q) {
    strt_dump_templ0(comm, base, name, q->nps, q->rr0_hst);
}

void rig_strt_dump(MPI_Comm comm, const char *base, const char *name, const int id, const RigQuants *q) {
    char code[FILENAME_MAX];
    gen_code(name, SS, code);
    restart_write_ss(comm, base, code, id, q->ns, q->ss_hst);
}

#undef PP
#undef SS






#include <stdio.h>
#include <vector_types.h>

#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)


#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)


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
struct Config;
struct float3;
struct int3;

void conf_ini(     Config **c);
void conf_fin(     Config *c);

void conf_read(int argc, char **argv,      Config *cfg);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);

bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);

void conf_write_exe(const Config *cfg, FILE *stream);
void conf_write_history(const Config *cfg, FILE *stream);

struct RigPinInfo;
struct int3;
struct Config;

#ifndef UDX_RIG_PERIODIC
#define UDX_RIG_PERIODIC
enum {NOT_PERIODIC = -1};
#endif

void rig_pininfo_ini(RigPinInfo **);
void rig_pininfo_fin(RigPinInfo *);
void rig_pininfo_set(int3 com, int3 axis, RigPinInfo *);
void rig_pininfo_set_pdir(int pdir, RigPinInfo *);
void rig_pininfo_set_conf(const Config *cfg, const char *ns, RigPinInfo *);

int rig_pininfo_get_pdir(const RigPinInfo *);

struct Particle;
struct Rigid;
struct Force;

void rig_reinit_ft(const int ns,      Rigid *ss);
void rig_update(const RigPinInfo *pi, float dt, int n, const Force *ff, const float *rr0, int ns,      Particle *pp, Rigid *ss);
void rig_generate(int ns, const Rigid *ss, int nps, const float *rr0,      Particle *pp);
void rig_update_mesh(float dt, int ns, const Rigid *ss, int nv, const float *vv,      Particle *pp);


void rig_pininfo_set_conf(const Config *cfg, const char *ns, RigPinInfo *pi) {
    int3 com, axis;
    int pdir;

    UC(conf_lookup_int3_ns(cfg, ns, "pin_com", &com));
    UC(conf_lookup_int3_ns(cfg, ns, "pin_axis", &axis));
    UC(conf_lookup_int_ns(cfg, ns, "pdir", &pdir));

    UC(rig_pininfo_set(com, axis, pi));
    UC(rig_pininfo_set_pdir(pdir, pi));
}



#include <stdio.h>
#include <assert.h>



#ifndef UDX_RIG_CORE_TYPE
#define UDX_RIG_CORE_TYPE
#ifndef UDX_RBC_RIG_TYPE_COMMON
#define UDX_RBC_RIG_TYPE_COMMON
#include "types.cuh"
#endif
#endif





#define D2D d::MemcpyDeviceToDevice
#define D2H d::MemcpyDeviceToHost
#define H2D d::MemcpyHostToDevice
#define H2H d::MemcpyHostToHost
#define A2A d::MemcpyDefault

#define cD2D(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), D2D))
#define cH2H(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), H2H))
#define cA2A(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), A2A))
#define cD2H(H, D, n) CC(d::Memcpy((H), (D), (n) * sizeof((H)[0]), D2H))
#define cH2D(D, H, n) CC(d::Memcpy((D), (H), (n) * sizeof((H)[0]), H2D))

#define aD2D(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), D2D))
#define aH2H(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), H2H))
#define aA2A(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), A2A))
#define aD2H(H, D, n) CC(d::MemcpyAsync((H), (D), (n) * sizeof((H)[0]), D2H))
#define aH2D(D, H, n) CC(d::MemcpyAsync((D), (H), (n) * sizeof((H)[0]), H2D))


#define Dfree(D)      CC(d::Free(D))
#define Dalloc(D, n)  CC(d::Malloc((void**)(D), (n) * sizeof(**(D))))


#define Dset(P, v, n) CC(d::Memset(P, v, (n)*sizeof(*(P))))
#define Dzero(P, n)   Dset(P, 0, n)


#define DsetA(P, v, n) CC(d::MemsetAsync(P, v, (n)*sizeof(*(P))))
#define DzeroA(P, n)   DsetA(P, 0, n)

#ifndef UDX_RIG_DEVICE_API_COMMON
#define UDX_RIG_DEVICE_API_COMMON
#ifndef UDX_RBC_RIG_DEVICE_API_COMMON
#define UDX_RBC_RIG_DEVICE_API_COMMON
#include "device.cuh"
#endif
#endif

void msg_ini(int rank);
void msg_print(const char *fmt, ...);
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)


#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)


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


#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}




#ifndef UDX_RBC_RIG_KL_COMMON
#define UDX_RBC_RIG_KL_COMMON
#include "launch.cuh"
#endif
#ifndef UDX_RBC_RIG_SHFL_COMMON
#define UDX_RBC_RIG_SHFL_COMMON
#define _I_ static __device__

template <typename T>
_I_ T shfl_down(T val, unsigned int width) {
    return __shfl_down_sync(0xFFFFFFFF, val, width);
}

template <typename T>
_I_ T shfl_up(T val, unsigned int width) {
    return __shfl_up_sync(0xFFFFFFFF, val, width);
}
template <typename T>
_I_ T shfl(T val, unsigned int width) {
    return __shfl_sync(0xFFFFFFFF, val, width);
}

#undef _I_
#endif
#ifndef UDX_RBC_RIG_ALGO_DEV_COMMON
#define UDX_RBC_RIG_ALGO_DEV_COMMON
#define _S_ static __device__
#define _I_ static __device__

template <typename T>
_I_ T warpReduceSum(T v)
{
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        v += shfl_down(v, offset);
    return v;
}

_I_ float2 warpReduceSum(float2 val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += shfl_down(val.x, offset);
        val.y += shfl_down(val.y, offset);
    }
    return val;
}

_I_ float3 warpReduceSum(float3 val)
{
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x += shfl_down(val.x, offset);
        val.y += shfl_down(val.y, offset);
        val.z += shfl_down(val.z, offset);
    }
    return val;
}

#undef _S_
#undef _I_

#endif



struct RigPinInfo {
    int3 com;
    int3 axis;
    int pdir;
};

static __device__ float rigid_dot(const float *v, const float *u) {
    enum {X, Y, Z};
    return v[X]*u[X] + v[Y]*u[Y] + v[Z]*u[Z];
}

static __device__ void rigid_reject(     float *v, float *u) {
    enum {X, Y, Z};
    const float d = rigid_dot(v, u);
    v[X] -= d*u[X]; v[Y] -= d*u[Y]; v[Z] -= d*u[Z];
}

static __device__ void rigid_normalize(     float *v) {
    enum {X, Y, Z};
    const float s = rsqrtf(rigid_dot(v, v));
    v[X] *= s; v[Y] *= s; v[Z] *= s;
}

__device__ void gram_schmidt(     float *e0, float *e1, float *e2) {

    rigid_normalize(e0);

    rigid_reject(e1, e0);
    rigid_normalize(e1);

    rigid_reject(e2, e0);
    rigid_reject(e2, e1);
    rigid_normalize(e2);
}

__device__ void rot_e(float dt, const float *om,      float *e) {
    enum {X, Y, Z};
    const float omx = om[X], omy = om[Y], omz = om[Z];
    const float ex = e[X], ey = e[Y], ez = e[Z];

    const float vx = omy*ez - omz*ey;
    const float vy = omz*ex - omx*ez;
    const float vz = omx*ey - omy*ex;

    e[X] += vx*dt; e[Y] += vy*dt; e[Z] += vz*dt;
}
__global__ void rigid_rot_referential_kernel(float dt, const int ns, Rigid *ss) {
    int i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < ns) {
        Rigid s = ss[i];
        rot_e(dt, s.om,      s.e0);
        rot_e(dt, s.om,      s.e1);
        rot_e(dt, s.om,      s.e2);

        gram_schmidt(     s.e0, s.e1, s.e2);
        ss[i] = s;
    }
}

static __device__ void rigid_atomic_add3(float d[3], float3 s) {
    enum {X, Y, Z};
    atomicAdd(d + X, s.x);
    atomicAdd(d + Y, s.y);
    atomicAdd(d + Z, s.z);
}

static __device__ float3 rigid_fetch_force(int i, const Force *ff) {
    enum {X, Y, Z};
    float3 f;
    f.x = ff[i].f[X];
    f.y = ff[i].f[Y];
    f.z = ff[i].f[Z];
    return f;
}

__global__ void rigid_add_f_to_kernel(const int nps, const Particle *pp, const Force *ff,      Rigid *ss) {
    enum {X, Y, Z};
    int gid, sid, i;
    gid = blockIdx.x * blockDim.x + threadIdx.x;
    sid = blockIdx.y;

    i = sid * nps + gid;

    float3 to, f = make_float3(0, 0, 0);
    Particle   p = {0, 0, 0, 0, 0, 0};

    if (gid < nps) {
        f = rigid_fetch_force(i, ff);
        p = pp[i];
    }

    const float dr[3] = {p.r[X] - ss[sid].com[X],
                         p.r[Y] - ss[sid].com[Y],
                         p.r[Z] - ss[sid].com[Z]};

    to.x = dr[Y] * f.z - dr[Z] * f.y;
    to.y = dr[Z] * f.x - dr[X] * f.z;
    to.z = dr[X] * f.y - dr[Y] * f.x;

    f =  warpReduceSum(f);
    to = warpReduceSum(to);

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        rigid_atomic_add3(ss[sid].fo, f);
        rigid_atomic_add3(ss[sid].to, to);
    }
}

__global__ void rigid_reinit_ft_kernel(const int ns, Rigid *ss) {
    enum {X, Y, Z};
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < ns) {
        Rigid *s = ss + gid;
        s->fo[X] = s->fo[Y] = s->fo[Z] = 0.f;
        s->to[X] = s->to[Y] = s->to[Z] = 0.f;
    }
}

__global__ void rigid_update_om_v_kernel(RigPinInfo pi, float dt, const int ns, Rigid *ss) {
    enum {X, Y, Z};
    enum {XX, XY, XZ, YY, YZ, ZZ};
    enum {YX = XY, ZX = XZ, ZY = YZ};
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < ns) {
        Rigid s = ss[gid];
        const float *A = s.Iinv, *b = s.to;
        const float sc = dt/s.mass;
        const float dom[3] = {A[XX]*b[X] + A[XY]*b[Y] + A[XZ]*b[Z],
                              A[YX]*b[X] + A[YY]*b[Y] + A[YZ]*b[Z],
                              A[ZX]*b[X] + A[ZY]*b[Y] + A[ZZ]*b[Z]};

        s.om[X] += dom[X]*dt;
        s.om[Y] += dom[Y]*dt;
        s.om[Z] += dom[Z]*dt;

        if (pi.axis.x) s.om[X] = 0;
        if (pi.axis.y) s.om[Y] = 0;
        if (pi.axis.z) s.om[Z] = 0;


        s.v[X] += s.fo[X] * sc;
        s.v[Y] += s.fo[Y] * sc;
        s.v[Z] += s.fo[Z] * sc;

        if (pi.com.x) s.v[X] = 0;
        if (pi.com.y) s.v[Y] = 0;
        if (pi.com.z) s.v[Z] = 0;

        ss[threadIdx.x] = s;
    }
}

__global__ void rigid_update_com_kernel(float dt, int ns, Rigid *ss) {
    int sid, c, i;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    sid = i / 3;
    c   = i % 3;

    if (sid < ns)
        ss[sid].com[c] += ss[sid].v[c]*dt;
}

__global__ void rigid_update_pp_kernel(const int nps, const float *rr0, const Rigid *ss,      Particle *pp) {
    enum {X, Y, Z};
    int pid, sid, i;
    Rigid s;
    float x, y, z, dx, dy, dz, omx, omy, omz;
    Particle p;
    const float *r0;
    pid = threadIdx.x + blockIdx.x * blockDim.x;
    sid = blockIdx.y;
    i = sid * nps + pid;

    s = ss[sid];
    omx = s.om[X]; omy = s.om[Y]; omz = s.om[Z];

    if (pid < nps) {
        p = pp[i];
        r0 = &rr0[3*pid];
        x = r0[X]; y = r0[Y]; z = r0[Z];

        dx = x * s.e0[X] + y * s.e1[X] + z * s.e2[X];
        dy = x * s.e0[Y] + y * s.e1[Y] + z * s.e2[Y];
        dz = x * s.e0[Z] + y * s.e1[Z] + z * s.e2[Z];

        p.v[X] = s.v[X] + omy * dz - omz * dy;
        p.v[Y] = s.v[Y] + omz * dx - omx * dz;
        p.v[Z] = s.v[Z] + omx * dy - omy * dx;

        p.r[X] = s.com[X] + dx;
        p.r[Y] = s.com[Y] + dy;
        p.r[Z] = s.com[Z] + dz;

        pp[i] = p;
    }
}

__global__ void rigid_update_mesh_kernel(float dt, const Rigid *ss_dev, const int nv, const float *vv,      Particle *pp) {
    enum {X, Y, Z};
    const int sid = blockIdx.y;
    const Rigid *s = ss_dev + sid;

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int vid = sid * nv + i;

    if (i < nv) {
        const float x = vv[3*i + X];
        const float y = vv[3*i + Y];
        const float z = vv[3*i + Z];

        const Particle p0 = pp[vid];
        Particle p;

        p.r[X] = x * s->e0[X] + y * s->e1[X] + z * s->e2[X] + s->com[X];
        p.r[Y] = x * s->e0[Y] + y * s->e1[Y] + z * s->e2[Y] + s->com[Y];
        p.r[Z] = x * s->e0[Z] + y * s->e1[Z] + z * s->e2[Z] + s->com[Z];

        p.v[X] = (p.r[X] - p0.r[X]) / dt;
        p.v[Y] = (p.r[Y] - p0.r[Y]) / dt;
        p.v[Z] = (p.r[Z] - p0.r[Z]) / dt;

        pp[vid] = p;
    }
}
enum {NTHRD=128};

void rig_pininfo_ini(RigPinInfo **pi) {
    RigPinInfo *p;
    EMALLOC(1, pi);
    p = *pi;
    p->pdir = NOT_PERIODIC;
}

void rig_pininfo_fin(RigPinInfo *pi) {
    EFREE(pi);
}

void rig_pininfo_set(int3 com, int3 axis, RigPinInfo *pi) {
    pi->com  = com;
    pi->axis = axis;
}

void rig_pininfo_set_pdir(int pdir, RigPinInfo *pi) {
    pi->pdir = pdir;
}

int rig_pininfo_get_pdir(const RigPinInfo *pi) {
    return pi->pdir;
}

void rig_reinit_ft(const int nsolid,      Rigid *ss) {
    KL(rigid_reinit_ft_kernel, (k_cnf(nsolid)), (nsolid,      ss));
}

static bool pinned_com(const RigPinInfo *pi) {
    int3 com = pi->com;
    return com.x && com.y && com.z;
}

void rig_update(const RigPinInfo *pi, float dt, int n, const Force *ff, const float *rr0, int ns,      Particle *pp, Rigid *ss) {
    if (ns < 1) return;

    const int nps = n / ns;

    const dim3 nblck ( ceiln(nps, NTHRD), ns );
    const dim3 nthrd ( NTHRD, 1 );

    KL(rigid_add_f_to_kernel, ( nblck, nthrd ), (nps, pp, ff,      ss));
    KL(rigid_update_om_v_kernel, (1, ns), (*pi, dt, ns,      ss));
    if (!pinned_com(pi))
        KL(rigid_update_com_kernel, (1, 3*ns ), (dt, ns,      ss));
    KL(rigid_rot_referential_kernel, (1, ns), (dt, ns,      ss));

    KL(rigid_update_pp_kernel, ( nblck, nthrd ), (nps, rr0, ss,      pp));
}

void rig_generate(int ns, const Rigid *ss, int nps, const float *rr0,      Particle *pp) {
    if (ns < 1) return;

    const dim3 nblck ( ceiln(nps, NTHRD), ns );
    const dim3 nthrd ( NTHRD, 1 );

    KL(rigid_update_pp_kernel, ( nblck, nthrd ), (nps, rr0, ss,      pp));
}

void rig_update_mesh(float dt, int ns, const Rigid *ss, int nv, const float *vv,      Particle *pp) {
    const dim3 nblck ( ceiln(nv, NTHRD), ns );
    const dim3 nthrd ( NTHRD, 1 );
    KL(rigid_update_mesh_kernel, (nblck, nthrd ), (dt, ss, nv, vv,      pp));
}












#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <mpi.h>




#ifndef UDX_RBC_CORE_DEF_COMMON
#define UDX_RBC_CORE_DEF_COMMON
#ifndef UDX_RBC_RIG_DEF_COMMON
#define UDX_RBC_RIG_DEF_COMMON
#include "constants.cuh"
#endif
#endif
#ifndef UDX_RBC_CORE_TYPE_COMMON
#define UDX_RBC_CORE_TYPE_COMMON
#ifndef UDX_RBC_RIG_TYPE_COMMON
#define UDX_RBC_RIG_TYPE_COMMON
#include "types.cuh"
#endif
#endif





#define D2D d::MemcpyDeviceToDevice
#define D2H d::MemcpyDeviceToHost
#define H2D d::MemcpyHostToDevice
#define H2H d::MemcpyHostToHost
#define A2A d::MemcpyDefault

#define cD2D(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), D2D))
#define cH2H(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), H2H))
#define cA2A(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), A2A))
#define cD2H(H, D, n) CC(d::Memcpy((H), (D), (n) * sizeof((H)[0]), D2H))
#define cH2D(D, H, n) CC(d::Memcpy((D), (H), (n) * sizeof((H)[0]), H2D))

#define aD2D(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), D2D))
#define aH2H(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), H2H))
#define aA2A(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), A2A))
#define aD2H(H, D, n) CC(d::MemcpyAsync((H), (D), (n) * sizeof((H)[0]), D2H))
#define aH2D(D, H, n) CC(d::MemcpyAsync((D), (H), (n) * sizeof((H)[0]), H2D))


#define Dfree(D)      CC(d::Free(D))
#define Dalloc(D, n)  CC(d::Malloc((void**)(D), (n) * sizeof(**(D))))


#define Dset(P, v, n) CC(d::Memset(P, v, (n)*sizeof(*(P))))
#define Dzero(P, n)   Dset(P, 0, n)


#define DsetA(P, v, n) CC(d::MemsetAsync(P, v, (n)*sizeof(*(P))))
#define DzeroA(P, n)   DsetA(P, 0, n)

void msg_ini(int rank);
void msg_print(const char *fmt, ...);

#ifndef UDX_RBC_CORE_DEVICE_API_COMMON
#define UDX_RBC_CORE_DEVICE_API_COMMON
#ifndef UDX_RBC_RIG_DEVICE_API_COMMON
#define UDX_RBC_RIG_DEVICE_API_COMMON
#include "device.cuh"
#endif
#endif
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)


#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)


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
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);


#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}






#define MC(ans)                                                         \
    do { mpicheck::check((ans), __FILE__, __LINE__); } while (0)

namespace mpicheck {
void check(int code, const char *file, int line);
}
namespace m {
int Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int Barrier(MPI_Comm comm);
int Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
int Cancel(MPI_Request *request);
int Cart_rank(MPI_Comm comm, const int coords[], int *rank);
int Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart);
int Cart_coords(MPI_Comm comm, int rank, int maxdims, int coords[]);
int Cart_get(MPI_Comm comm, int maxdims, int dims[], int periods[], int coords[]);
int Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
int Comm_free(MPI_Comm *comm);
int Comm_rank(MPI_Comm comm, int *rank);
int Comm_size( MPI_Comm comm, int *size );
int Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
int Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
int File_close(MPI_File *fh);
int File_get_position(MPI_File fh, MPI_Offset *offset);
int File_open(MPI_Comm comm, const char *filename, int amode, MPI_Info info, MPI_File *fh);
int File_seek(MPI_File fh, MPI_Offset offset, int whence);
int File_set_size(MPI_File fh, MPI_Offset size);
int File_write_at_all(MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);
int Finalize(void);
int Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
           int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
int Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
            const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm);
int Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
              int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
int Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
               const int displs[], MPI_Datatype recvtype, MPI_Comm comm);
int Init(int *argc, char ***argv);
int Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
int Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
int Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);
int Type_commit(MPI_Datatype *type);
int Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
int Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses);
int Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
int Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);
int Error_string(int errorcode, char *string, int *resultlen);


bool is_master(MPI_Comm comm);
int is_success(int errorcode);
int is_pending(int errorcode);
int is_err_in_status(int errorcode);
int status2errcode(MPI_Status *status);
}

struct int4;
struct MeshRead;

void mesh_read_ini_off(const char *path, MeshRead**);
void mesh_read_ini_ply(const char *path, MeshRead**);
void mesh_read_fin(MeshRead*);

int mesh_read_get_nt(const MeshRead*);
int mesh_read_get_nv(const MeshRead*);
int mesh_read_get_ne(const MeshRead*);
int mesh_read_get_md(const MeshRead*);

const int4  *mesh_read_get_tri(const MeshRead*);
const float *mesh_read_get_vert(const MeshRead*);
const int4  *mesh_read_get_dih(MeshRead*);
#ifndef UDX_RBC_RIG_RESTART_COMMON
#define UDX_RBC_RIG_RESTART_COMMON
struct Coords;
struct Particle;
struct Rigid;

enum {RESTART_TEMPL=-1,
      RESTART_FINAL=-2};
enum {RESTART_BEGIN=RESTART_FINAL};

typedef void (*StreamWriter)(const void*, FILE*);
typedef void (*StreamReader)(FILE*, void*);

void restart_write_pp(MPI_Comm, const char *base, const char *code, int id, long n, const Particle *pp);
void restart_write_ii(MPI_Comm, const char *base, const char *code, int id, long n, const int *ii);
void restart_write_ss(MPI_Comm, const char *base, const char *code, int id, long n, const Rigid *ss);
void restart_write_stream_one_node(MPI_Comm, const char *base, const char *code, int id, const void *data, StreamWriter);

void restart_read_pp(MPI_Comm, const char *base, const char *code, int id, int *n, Particle *pp);
void restart_read_ii(MPI_Comm, const char *base, const char *code, int id, int *n, int *ii);
void restart_read_ss(MPI_Comm, const char *base, const char *code, int id, int *n, Rigid *ss);
void restart_read_stream_one_node(const char *base, const char *code, int id, StreamReader, void *data);
#endif

struct AreaVolume;
struct Particle;

struct RbcQuants {
    int n, nc;
    int nt, nv;
    int md;
    Particle *pp, *pp_hst;
    bool ids;
    int *ii;
    AreaVolume *area_volume;
};

#ifndef UDX_RBC_ADJ_MAP_TYPE
#define UDX_RBC_ADJ_MAP_TYPE
struct AdjMap {
    int i0, i1, i2, i3, i4;
    int rbc;
};
#endif
struct Adj;
struct Adj_v;
struct AdjMap;
struct int4;

void adj_ini(int md, int nt, int nv, const int4 *tt,      Adj**);
void adj_fin(Adj*);
int  adj_get_map(int i, const Adj*,      AdjMap *m);
int  adj_get_max(const Adj*);
void adj_get_anti(const Adj*,      int *anti);

void adj_view_ini(const Adj*, Adj_v**);
void adj_view_fin(Adj_v*);
struct Coords;
struct Particle;
struct Matrices;


int  mesh_gen_from_file(const Coords*, const float *vv, const char *ic, int nv,      Particle*);
void mesh_gen_from_matrices(int nv, const float *vv, const Matrices*,      int *pn, Particle*);
void mesh_shift(const Coords*, int n,      Particle*);
struct RbcShape;
struct MeshRead;
struct Adj;



void rbc_shape_ini(const Adj*, const float *rr,      RbcShape**);
void rbc_shape_fin(     RbcShape*);

void rbc_shape_edg (RbcShape*,      float**);
void rbc_shape_area(RbcShape*,      float**);

void rbc_shape_total_area(RbcShape*,      float*);
void rbc_shape_total_volume(RbcShape*,      float*);
struct AreaVolume;
struct Particle;
struct int4;

void area_volume_ini(int nv, int nt, const int4 *tt, int max_cell, AreaVolume**);
void area_volume_fin(AreaVolume*);

void area_volume_compute(AreaVolume*, int nm, const Particle*,      float **av);
void area_volume_host(AreaVolume*,      float **av);
struct Coords;
struct MeshRead;

void rbc_ini(long maxnc, bool ids, const MeshRead*, RbcQuants*);
void rbc_fin(RbcQuants*);

void rbc_gen_mesh(const Coords*, MPI_Comm, const MeshRead*, const char *ic,      RbcQuants*);
void rbc_gen_freeze(MPI_Comm,      RbcQuants*);
void rbc_strt_quants(MPI_Comm, const MeshRead*, const char *base, const char *name, const int id, RbcQuants*);
void rbc_strt_dump(MPI_Comm, const char *base, const char *name, int id, const RbcQuants *q);


static void ini_common(long maxnc, const int4 *tt, RbcQuants *q) {
    int nv, nt;
    nv = q->nv; nt = q->nt;
    Dalloc(&q->pp, maxnc * nv);
    EMALLOC(maxnc * nv, &q->pp_hst);
    UC(area_volume_ini(nv, nt, tt, maxnc,      &q->area_volume));
}

static void ini_ids(long maxnc, RbcQuants *q)  { EMALLOC(maxnc, &q->ii); }

void rbc_ini(long maxnc, bool ids, const MeshRead *cell, RbcQuants *q) {
    const int4 *tt;
    q->nv = mesh_read_get_nv(cell);
    q->nt = mesh_read_get_nt(cell);
    q->md = mesh_read_get_md(cell);
    tt = mesh_read_get_tri(cell);

    q->n = q->nc = 0;
    q->ids = ids;
    UC(ini_common(maxnc, tt, q));
    if (ids) UC(ini_ids(maxnc, q));
}
static void fin_common(RbcQuants *q) {
    Dfree(q->pp);
    UC(area_volume_fin(q->area_volume));
    EFREE(q->pp_hst);
}

static void fin_ids(RbcQuants *q) { EFREE(q->ii);   }

void rbc_fin(RbcQuants *q) {
    if (q->ids) fin_ids(q);
    fin_common(q);
}
static void setup_from_pos(const Coords *coords, MPI_Comm comm, const float *vv, const char *ic, int nv,
                           Particle *pp, int *pnc, int *pn,               Particle *pp_hst) {
    int nc;
    nc = mesh_gen_from_file(coords, vv, ic, nv,      pp_hst);
    if (nc >= MAX_CELL_NUM)
        ERR("nc=%d >= MAX_CELL_NUM=%d", nc, MAX_CELL_NUM);
    if (nc) cH2D(pp, pp_hst, nv * nc);
    MC(m::Barrier(comm));
    *pnc = nc; *pn = nc * nv;
}

static void gen_ids(MPI_Comm comm, long nc,      int *ii) {
    long i, i0 = 0, count = 1;
    MC(m::Exscan(&nc, &i0, count, MPI_LONG, MPI_SUM, comm));
    for (i = 0; i < nc; ++i)
        ii[i] = i + i0;
}

void rbc_gen_mesh(const Coords *coords, MPI_Comm comm, const MeshRead *off, const char *ic,      RbcQuants *q) {
    const float *vv;
    vv = mesh_read_get_vert(off);
    setup_from_pos(coords, comm, vv, ic, q->nv,      q->pp, &q->nc, &q->n,       q->pp_hst);
}

void rbc_gen_freeze(MPI_Comm comm,      RbcQuants *q) {
    if (q->ids) UC(gen_ids(comm, q->nc,      q->ii));
}
#define PP  ".pp"
#define IDS ".ids"

static void rbc_gen_code(const char *name, const char *ext, char *code) {
    strcpy(code, name);
    strcat(code, ext);
}

static void setup_from_strt(MPI_Comm comm, const char *base, const char *name, int nv, int id,      Particle *pp, int *nc, int *n,       Particle *pp_hst) {
    char code[FILENAME_MAX];
    rbc_gen_code(name, PP, code);
    restart_read_pp(comm, base, code, id, n, pp_hst);
    *nc = *n / nv;

    if (*n) cH2D(pp, pp_hst, *n);
}

static void ids_from_strt(MPI_Comm comm, const char *base, const char *name, int id,      int *ii) {
    int nc;
    char code[FILENAME_MAX];
    rbc_gen_code(name, IDS, code);
    restart_read_ii(comm, base, code, id, &nc, ii);
}

void rbc_strt_quants(MPI_Comm comm, const MeshRead *off, const char *base, const char *name, int id, RbcQuants *q) {
    int nv;
    nv = mesh_read_get_nv(off);
    setup_from_strt(comm, base, name, nv, id,      q->pp, &q->nc, &q->n,       q->pp_hst);
    if (q->ids) ids_from_strt(comm, base, name, id,      q->ii);
}

static void strt_dump(MPI_Comm comm, const char *base, const char *name, int id, int n, const Particle *pp,       Particle *pp_hst) {
    char code[FILENAME_MAX];
    rbc_gen_code(name, PP, code);
    if (n) cD2H(pp_hst, pp, n);
    restart_write_pp(comm, base, code, id, n, pp_hst);
}

static void strt_dump_ii(MPI_Comm comm, const char *base, const char *name, int id, int nc, const int *ii) {
    char code[FILENAME_MAX];
    rbc_gen_code(name, IDS, code);
    restart_write_ii(comm, base, code, id, nc, ii);
}

void rbc_strt_dump(MPI_Comm comm, const char *base, const char *name, int id, const RbcQuants *q) {
    strt_dump(comm, base, name, id, q->n, q->pp,       q->pp_hst);
    if (q->ids)
        strt_dump_ii(comm, base, name, id, q->nc, q->ii);
}

#undef PP
#undef IDS











#include <stdio.h>




#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)


#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)


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
struct Config;
struct float3;
struct int3;

void conf_ini(     Config **c);
void conf_fin(     Config *c);

void conf_read(int argc, char **argv,      Config *cfg);

void conf_lookup_int(const Config *c, const char *desc, int *a);
void conf_lookup_float(const Config *c, const char *desc, float *a);
void conf_lookup_bool(const Config *c, const char *desc, int *a);
void conf_lookup_string(const Config *c, const char *desc, const char **a);
void conf_lookup_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3(const Config *c, const char *desc, int3 *a);
void conf_lookup_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3(const Config *c, const char *desc, float3 *a);
void conf_lookup_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_lookup_int_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_float_ns(const Config *c, const char *ns, const char *desc, float *a);
void conf_lookup_bool_ns(const Config *c, const char *ns, const char *desc, int *a);
void conf_lookup_string_ns(const Config *c, const char *ns, const char *desc, const char **a);
void conf_lookup_vint_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, int a[]);
void conf_lookup_int3_ns(const Config *c, const char *ns, const char *desc, int3 *a);
void conf_lookup_vfloat_ns(const Config *c, const char *ns, const char *desc, int maxn, int *n, float a[]);
void conf_lookup_float3_ns(const Config *c, const char *ns, const char *desc, float3 *a);
void conf_lookup_vstring_ns(const Config *c, const char *ns, const char *d, int maxn, int *n, const char **ss);

bool conf_opt_int(const Config *c, const char *desc, int *a);
bool conf_opt_float(const Config *c, const char *desc, float *a);
bool conf_opt_bool(const Config *c, const char *desc, int *a);
bool conf_opt_string(const Config *c, const char *desc, const char **a);
bool conf_opt_vint(const Config *c, const char *desc, int maxn, int *n, int a[]);
bool conf_opt_int3(const Config *c, const char *desc, int3 *a);
bool conf_opt_vfloat(const Config *c, const char *desc, int maxn, int *n, float a[]);
bool conf_opt_float3(const Config *c, const char *desc, float3 *a);
bool conf_opt_vstring(const Config *c, const char *desc, int maxn, int *n, const char **ss);

void conf_set_int(const char *desc, int a, Config *c);
void conf_set_vint(const char *desc, int nelem, const int a[], Config *cfg);
void conf_set_int3(const char *desc, int3 a, Config *cfg);
void conf_set_float(const char *desc, float a, Config *cfg);
void conf_set_vfloat(const char *desc, int nelem, const float a[], Config *cfg);
void conf_set_float3(const char *desc, float3 a, Config *cfg);
void conf_set_bool(const char *desc, int a, Config *cfg);
void conf_set_string(const char *desc, const char *a, Config *cfg);

void conf_write_exe(const Config *cfg, FILE *stream);
void conf_write_history(const Config *cfg, FILE *stream);

struct Config;
struct RbcParams;
struct RbcParams_v {
    float gammaC, gammaT, kBT;
    float kb, phi;
    float ks, x0, mpow;
    float ka, kd, kv;
    float totArea, totVolume;
};

void rbc_params_ini(RbcParams **);
void rbc_params_fin(RbcParams *);

void rbc_params_set_fluct(float gammaC, float gammaT, float kBT, RbcParams *);
void rbc_params_set_bending(float kb, float phi, RbcParams *);
void rbc_params_set_spring(float ks, float x0, float mpow , RbcParams *);
void rbc_params_set_area_volume(float ka, float kd, float kv, RbcParams *);
void rbc_params_set_tot_area_volume(float totArea, float totVolume, RbcParams *);

double rbc_params_get_tot_volume(const RbcParams *);

void rbc_params_set_conf(const Config *c, const char *name, RbcParams *);

RbcParams_v rbc_params_get_view(const RbcParams *);

void rbc_params_set_conf(const Config *c, const char *name, RbcParams *par) {
    float gc, gt, kbt, kb, phi, ks, mpow, x0, ka, kd, kv, totArea, totVolume;

    UC(conf_lookup_float_ns(c, name, "gammaC", &gc));
    UC(conf_lookup_float_ns(c, name, "gammaT", &gt));
    UC(conf_lookup_float_ns(c, name, "kBT", &kbt));
    UC(conf_lookup_float_ns(c, name, "kb", &kb));
    UC(conf_lookup_float_ns(c, name, "phi", &phi));
    UC(conf_lookup_float_ns(c, name, "ks", &ks));
    UC(conf_lookup_float_ns(c, name, "x0", &x0));
    UC(conf_lookup_float_ns(c, name, "mpow", &mpow));
    UC(conf_lookup_float_ns(c, name, "ka", &ka));
    UC(conf_lookup_float_ns(c, name, "kd", &kd));
    UC(conf_lookup_float_ns(c, name, "kv", &kv));

    UC(conf_lookup_float_ns(c, name, "totArea",   &totArea));
    UC(conf_lookup_float_ns(c, name, "totVolume", &totVolume));

    rbc_params_set_fluct(gc, gt, kbt,      par);
    rbc_params_set_bending(kb, phi,      par);
    rbc_params_set_spring(ks, x0, mpow,      par);
    rbc_params_set_area_volume(ka, kd, kv,      par);
    rbc_params_set_tot_area_volume(totArea, totVolume,      par);
}



#include <stdlib.h>
#include <stdio.h>

void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)


#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)


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


struct RbcParams {
    float gammaC;
    float gammaT;
    float kBT;
    float kb;
    float phi;
    float ks;
    float x0;
    float mpow;
    float ka;
    float kd;
    float kv;


    float totVolume;
    float totArea;
};


void rbc_params_ini(RbcParams **p) {
    EMALLOC(1, p);
}

void rbc_params_fin(RbcParams *p) {
    EFREE(p);
}

void rbc_params_set_fluct(float gammaC, float gammaT, float kBT, RbcParams *par) {
    par->gammaC = gammaC;
    par->gammaT = gammaT;
    par->kBT = kBT;
}

void rbc_params_set_bending(float kb, float phi, RbcParams *par) {
    par->kb = kb;
    par->phi = phi;
}

void rbc_params_set_spring(float ks, float x0, float mpow, RbcParams *par) {
    par->ks = ks;
    par->x0 = x0;
    par->mpow = mpow;
}

void rbc_params_set_tot_area_volume(float totArea, float totVolume, RbcParams *par) {
    par->totArea = totArea;
    par->totVolume = totVolume;
}

void rbc_params_set_area_volume(float ka, float kd, float kv, RbcParams *par) {
    par->ka = ka;
    par->kv = kv;
    par->kd = kd;
}

double rbc_params_get_tot_volume(const RbcParams *par) {
    return par->totVolume;
}

RbcParams_v rbc_params_get_view(const RbcParams *p) {
    RbcParams_v v;
    v.gammaC = p->gammaC;
    v.gammaT = p->gammaT;
    v.kBT = p->kBT;
    v.kb = p->kb;
    v.phi = p->phi;
    v.ks = p->ks;
    v.x0 = p->x0;
    v.mpow = p->mpow;
    v.ka = p->ka;
    v.kd = p->kd;
    v.kv = p->kv;
    v.totArea = p->totArea;
    v.totVolume = p->totVolume;
    return v;
}





#include <stdio.h>
#include <math.h>

#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)


#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)


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
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);
void msg_ini(int rank);
void msg_print(const char *fmt, ...);

struct Adj;
struct Adj_v;
struct AdjMap;
struct int4;

void adj_ini(int md, int nt, int nv, const int4 *tt,      Adj**);
void adj_fin(Adj*);
int  adj_get_map(int i, const Adj*,      AdjMap *m);
int  adj_get_max(const Adj*);
void adj_get_anti(const Adj*,      int *anti);

void adj_view_ini(const Adj*, Adj_v**);
void adj_view_fin(Adj_v*);
#ifndef UDX_RBC_ADJ_MAP_TYPE
#define UDX_RBC_ADJ_MAP_TYPE
struct AdjMap {
    int i0, i1, i2, i3, i4;
    int rbc;
};
#endif
namespace tri_hst {
double kahan_area0(double a, double b, double c);
double kahan_area(const double a[3], const double b[3], const double c[3]);
void   ac_bc_cross(const double a[3], const double b[3], const double c[3],      double r[3]);
double shewchuk_area(const double a[3], const double b[3], const double c[3]);
double orient3d(const double a[3], const double b[3], const double c[3], const double d[3]);
void dihedral_xy(const double a[3], const double b[3], const double c[3], const double d[3],
                   double *x, double *y);
}

struct RbcShape;
struct MeshRead;
struct Adj;



void rbc_shape_ini(const Adj*, const float *rr,      RbcShape**);
void rbc_shape_fin(     RbcShape*);

void rbc_shape_edg (RbcShape*,      float**);
void rbc_shape_area(RbcShape*,      float**);

void rbc_shape_total_area(RbcShape*,      float*);
void rbc_shape_total_volume(RbcShape*,      float*);

struct RbcShape {
    float *edg, *area;
    float A, V;
};
static void diff(const float *a, const float *b,      float *c) {
    enum {X, Y, Z};
    c[X] = a[X] - b[X];
    c[Y] = a[Y] - b[Y];
    c[Z] = a[Z] - b[Z];
}
static double vabs(float *a) {
    enum {X, Y, Z};
    double r;
    r = a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z];
    return sqrt(r);
}
static void compute_edg(const Adj *adj, const float *rr,      float *o) {
    int i, valid, n;
    const float *r0, *r1;
    float r01[3];
    float a;
    AdjMap m;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj,      &m);
        if (!valid) continue;
        r0 = &rr[3*m.i0]; r1 = &rr[3*m.i1];
        diff(r0, r1,      r01);
        a = vabs(r01);
        o[i] = a;
    }
}

static void copy(const float f[3], double d[3]) {
    enum {X, Y, Z};
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static double area(const float af[3], const float bf[3], const float cf[3]) {
    double ad[3], bd[3], cd[3];
    copy(af, ad); copy(bf, bd); copy(cf, cd);
    return tri_hst::kahan_area(ad, bd, cd);
}
static void compute_area(const Adj *adj, const float *rr,      float *o) {
    int i, i0, i1, i2, valid, n;
    const float *r0, *r1, *r2;
    AdjMap m;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj,      &m);
        if (!valid) continue;
        i0 = m.i0; i1 = m.i1; i2 = m.i2;
        r0 = &rr[3*i0]; r1 = &rr[3*i1]; r2 = &rr[3*i2];
        o[i] = area(r0, r1, r2);
    }
}

static void compute_total_volume(const Adj *adj, const float *rr,      float *pV) {
    enum {X, Y, Z};
    int i, i0, i1, i2, valid, n;
    const float *r0, *r1, *r2;
    float x0, y0, z0, x1, y1, z1, x2, y2, z2;
    AdjMap m;
    double V;
    V = 0;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj,      &m);
        if (!valid) continue;
        i0 = m.i0; i1 = m.i1; i2 = m.i2;
        r0 = &rr[3*i0]; r1 = &rr[3*i1]; r2 = &rr[3*i2];
        x0 = r0[X]; y0 = r0[Y]; z0 = r0[Z];
        x1 = r1[X]; y1 = r1[Y]; z1 = r1[Z];
        x2 = r2[X]; y2 = r2[Y]; z2 = r2[Z];
        V += (x0*y1-x1*y0)*z2+x2*(y0*z1-y1*z0)+y2*(x1*z0-x0*z1);
    }
    V /= 6;
    V /= 3;
    msg_print("V: %g", V);
    *pV = V;
}

static void compute_total_area(const Adj *adj, const float* area,      float *pA) {
    int n, i, valid;
    AdjMap m;
    double A;

    A = 0;
    n = adj_get_max(adj);
    for (i = 0; i < n; i++) {
        valid = adj_get_map(i, adj,      &m);
        if (!valid) continue;
        A += area[i];
    }
    A /= 3;
    *pA = A;
}

void rbc_shape_ini(const Adj *adj, const float *rr,      RbcShape **pq) {
    int n;
    RbcShape *q;
    n = adj_get_max(adj);
    EMALLOC(1, &q);
    EMALLOC(n, &q->edg);
    EMALLOC(n, &q->area);

    compute_edg(adj, rr,      q->edg);
    compute_area(adj, rr,      q->area);
    compute_total_area(adj, q->area,      &q->A);
    compute_total_volume(adj, rr,      &q->V);

    msg_print("A: %g", q->A);

    *pq = q;
}

void rbc_shape_fin(RbcShape *q) {
    EFREE(q->edg);
    EFREE(q->area);
    EFREE(q);
}

void rbc_shape_edg  (RbcShape *q,      float** pe) { *pe = q->edg; }
void rbc_shape_area (RbcShape *q,      float** pe) { *pe = q->area; }

void rbc_shape_total_area(RbcShape *q,      float *pe)   { *pe = q->A; }
void rbc_shape_total_volume(RbcShape *q,      float *pe) {
    ERR("rbc_shape_total_volume is not implimented");
    *pe = q->V;
}






#include <stdio.h>
#include <assert.h>




#ifndef UDX_RBC_CORE_DEF_COMMON
#define UDX_RBC_CORE_DEF_COMMON
#ifndef UDX_RBC_RIG_DEF_COMMON
#define UDX_RBC_RIG_DEF_COMMON
#include "constants.cuh"
#endif
#endif
#ifndef UDX_RBC_CORE_TYPE_COMMON
#define UDX_RBC_CORE_TYPE_COMMON
#ifndef UDX_RBC_RIG_TYPE_COMMON
#define UDX_RBC_RIG_TYPE_COMMON
#include "types.cuh"
#endif
#endif





#define D2D d::MemcpyDeviceToDevice
#define D2H d::MemcpyDeviceToHost
#define H2D d::MemcpyHostToDevice
#define H2H d::MemcpyHostToHost
#define A2A d::MemcpyDefault

#define cD2D(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), D2D))
#define cH2H(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), H2H))
#define cA2A(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), A2A))
#define cD2H(H, D, n) CC(d::Memcpy((H), (D), (n) * sizeof((H)[0]), D2H))
#define cH2D(D, H, n) CC(d::Memcpy((D), (H), (n) * sizeof((H)[0]), H2D))

#define aD2D(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), D2D))
#define aH2H(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), H2H))
#define aA2A(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), A2A))
#define aD2H(H, D, n) CC(d::MemcpyAsync((H), (D), (n) * sizeof((H)[0]), D2H))
#define aH2D(D, H, n) CC(d::MemcpyAsync((D), (H), (n) * sizeof((H)[0]), H2D))


#define Dfree(D)      CC(d::Free(D))
#define Dalloc(D, n)  CC(d::Malloc((void**)(D), (n) * sizeof(**(D))))


#define Dset(P, v, n) CC(d::Memset(P, v, (n)*sizeof(*(P))))
#define Dzero(P, n)   Dset(P, 0, n)


#define DsetA(P, v, n) CC(d::MemsetAsync(P, v, (n)*sizeof(*(P))))
#define DzeroA(P, n)   DsetA(P, 0, n)

void msg_ini(int rank);
void msg_print(const char *fmt, ...);



#if   defined(DEV_CUDA)


#elif defined(DEV_CPU)

#ifdef __device__
  #undef  __device__
  #define __device__
#endif

#ifdef __global__
  #undef  __global__
  #define __global__
#endif

#ifdef __host__
  #undef  __host__
  #define __host__
#endif

#ifdef __noinline__
  #undef  __noinline__
  #define __noinline__
#endif

#ifdef __forceinline__
  #undef  __forceinline__
  #define __forceinline__
#endif



#ifdef __constant__
  #undef  __constant__
  #define __constant__
#endif

#ifdef __shared__
  #undef  __shared__
  #define __shared__
#endif

#ifdef __managed__
  #undef  __managed__
  #define __managed__
#endif

#ifdef __restrict__
  #undef  __restrict__
  #define __restrict__
#endif
#else
  #error DEV_* is undefined
#endif
#ifndef UDX_RBC_CORE_DEVICE_API_COMMON
#define UDX_RBC_CORE_DEVICE_API_COMMON
#ifndef UDX_RBC_RIG_DEVICE_API_COMMON
#define UDX_RBC_RIG_DEVICE_API_COMMON
#include "device.cuh"
#endif
#endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
#define atomicAdd(...) (0)

#define atomicExch(...)
#define __float2half_rn(...) (0)
#define __fmaf_rz(...) (0.0)
#define __ldg(p) (*(p))
#define __popc(...)
#define __shfl(...) (0.0)
#define __shfl_down(...) (0.0)
#define __shfl_up(...) (0)
#define __shfl_xor(...) (0)

#define __syncthreads(...)

#define tex3D(...)

#define  Ifetch(t, i)  (0);
#define F4fetch(t, i)  (make_float4(0, 0, 0, 0))
#define F2fetch(t, i)  (make_float2(0, 0))
#define Ffetch(t, i)   (0.0)
#define Tfetch(T, t, i) (T{})

#define Ttex3D(T, to, i, j, k) (T{})
namespace d {
static uint3 threadIdx, blockIdx, gridDim;
static dim3  blockDim;
static int   warpSize = 1;
}

#define threadIdx d::threadIdx
#define blockDim  d::blockDim
#define gridDim   d::gridDim
#define blockIdx  d::blockIdx
#define warpSize  d::warpSize
#else
  #error DEV_* is undefined
#endif
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)


#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)


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
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);


#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}




#ifndef UDX_RBC_RIG_KL_COMMON
#define UDX_RBC_RIG_KL_COMMON
#ifndef UDX_RBC_CORE_KL_COMMON
#define UDX_RBC_CORE_KL_COMMON
#include "launch.cuh"
#endif
#endif

struct RbcStretch;
struct Force;

void rbc_stretch_ini(const char *path, int nv,      RbcStretch**);
void rbc_stretch_fin(RbcStretch*);
void rbc_stretch_apply(int nm, const RbcStretch*,      Force*);
static __device__ void rbc_stretch_apply0(const float *f,      float *fb) {
    enum {X, Y, Z};
    fb[X] += f[X]; fb[Y] += f[Y]; fb[Z] += f[Z];
}

static __device__ void rbc_stretch_apply1(int i, int nv, const float *f, Force *ff) {
    int k;
    k = i % nv;
    rbc_stretch_apply0(&f[3*k],      ff[i].f);
}

__global__ void rbc_stretch_apply_kernel(int n, int nv, const float *f, Force *ff) {
    int i;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;
    rbc_stretch_apply1(i, nv, f,      ff);
}struct RbcStretch {
    int nv;
    float *f;
};
static void alloc(int n, RbcStretch *f) { Dalloc(&f->f, 3*n); }
static void dealloc(RbcStretch *f) { Dfree(f->f); }

static int read3(FILE *f, float *h) {
    enum {X, Y, Z};
    int n;
    n = fscanf(f, "%f %f %f\n", &h[X], &h[Y], &h[Z]);
    return n;
}

static void read(FILE *f, int n,      float *h) {
    int i;
    i = 0;
    for (     ; i < n && read3(f, h) == 3;   i++, h += 3)
        ;
    if (i != n) ERR("got %d != %d lines", i, n);
}

static void ini0(const char* path, int n, float *h,      float *d) {
    FILE *f;
    msg_print("read '%s'", path);
    UC(efopen(path, "r",      &f));
    read(f, n ,      h);
    cH2D(d, h, 3*n);
    UC(efclose(f));
}

static void ini1(const char* path, int n,      RbcStretch *f) {
    float *d, *h;
    alloc(n, f);
    d = f->f;
    EMALLOC(3 * n, &h);
    UC(ini0(path, n,       h,      d));
    EFREE(h);
}

void rbc_stretch_ini(const char* path, int nv,      RbcStretch **fp) {
    RbcStretch *f;
    EMALLOC(1, &f);
    UC(ini1(path, nv, f));
    f->nv = nv;
    *fp = f;
}

void rbc_stretch_fin(RbcStretch *f) {
    dealloc(f);
    EFREE(f);
}

void rbc_stretch_apply(int nm, const RbcStretch *f,      Force *ff) {
    int n, nv;
    nv = f->nv;
    n  = nm * nv;
    if (n) KL(rbc_stretch_apply_kernel, (k_cnf(n)), (n, nv, f->f,      ff));
}










#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector_types.h>




#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)


#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)


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
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);

void msg_ini(int rank);
void msg_print(const char *fmt, ...);

#ifndef UDX_RBC_DEVICE_API_COMMON
#define UDX_RBC_DEVICE_API_COMMON
#ifndef UDX_RBC_RIG_DEVICE_API_COMMON
#define UDX_RBC_RIG_DEVICE_API_COMMON
#ifndef UDX_RBC_CORE_DEVICE_API_COMMON
#define UDX_RBC_CORE_DEVICE_API_COMMON
#include "device.cuh"
#endif
#endif
#endif


#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}









#define D2D d::MemcpyDeviceToDevice
#define D2H d::MemcpyDeviceToHost
#define H2D d::MemcpyHostToDevice
#define H2H d::MemcpyHostToHost
#define A2A d::MemcpyDefault

#define cD2D(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), D2D))
#define cH2H(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), H2H))
#define cA2A(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), A2A))
#define cD2H(H, D, n) CC(d::Memcpy((H), (D), (n) * sizeof((H)[0]), D2H))
#define cH2D(D, H, n) CC(d::Memcpy((D), (H), (n) * sizeof((H)[0]), H2D))

#define aD2D(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), D2D))
#define aH2H(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), H2H))
#define aA2A(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), A2A))
#define aD2H(H, D, n) CC(d::MemcpyAsync((H), (D), (n) * sizeof((H)[0]), D2H))
#define aH2D(D, H, n) CC(d::MemcpyAsync((D), (H), (n) * sizeof((H)[0]), H2D))


#define Dfree(D)      CC(d::Free(D))
#define Dalloc(D, n)  CC(d::Malloc((void**)(D), (n) * sizeof(**(D))))


#define Dset(P, v, n) CC(d::Memset(P, v, (n)*sizeof(*(P))))
#define Dzero(P, n)   Dset(P, 0, n)


#define DsetA(P, v, n) CC(d::MemsetAsync(P, v, (n)*sizeof(*(P))))
#define DzeroA(P, n)   DsetA(P, 0, n)

struct Adj;
struct Adj_v;
struct AdjMap;
struct int4;

void adj_ini(int md, int nt, int nv, const int4 *tt,      Adj**);
void adj_fin(Adj*);
int  adj_get_map(int i, const Adj*,      AdjMap *m);
int  adj_get_max(const Adj*);
void adj_get_anti(const Adj*,      int *anti);

void adj_view_ini(const Adj*, Adj_v**);
void adj_view_fin(Adj_v*);

void edg_ini(int md, int nv,                     int *hx);
void edg_set(int md, int i, int j, int val,      int *hx,       int *hy);
int  edg_get(int md, int i, int j,         const int *hx, const int *hy);
int  edg_valid(int md, int i, int j,         const int *hx);

#ifndef UDX_RBC_ADJ_MAP_TYPE
#define UDX_RBC_ADJ_MAP_TYPE
struct AdjMap {
    int i0, i1, i2, i3, i4;
    int rbc;
};
#endif
struct Adj_v {
    int *adj0;
    int *adj1;
    int nv, md;
};


struct Adj {
    int *adj0;
    int *adj1;
    int nv, md;
};
static void gen_a12(int md, int i0, int *hx, int *hy,      int *a1, int *a2) {
    int lo = i0*md, hi = lo + md, mi = hx[lo];
    int i;
    for (i = lo + 1; (i < hi) && (hx[i] != -1); i++)
        if (hx[i] < mi) mi = hx[i];

    int c = mi, c0;
    i = lo;
    do {
        c     = edg_get(md, i0, c0 = c, hx, hy);
        a1[i] = c0;
        a2[i] = edg_get(md, c, c0, hx, hy);
        i++;
    }  while (c != mi);
}

static void ini0(int md, int nt, int nv, const int4 *faces,      int *a1, int *a2) {
    int hx[nv*md], hy[nv*md];
    int i;
    for (i = 0; i < nv*md; i++) a1[i] = a2[i] = -1;
    edg_ini(nv, md, hx);

    int4 t;
    for (int ifa = 0; ifa < nt; ifa++) {
        t = faces[ifa];
        int f0 = t.x, f1 = t.y, f2 = t.z;
        edg_set(md, f0, f1, f2,   hx, hy);
        edg_set(md, f1, f2, f0,   hx, hy);
        edg_set(md, f2, f0, f1,   hx, hy);
    }
    for (i = 0; i < nv; i++) gen_a12(md, i, hx, hy,      a1, a2);
}

void adj_ini(int md, int nt, int nv, const int4 *faces,      Adj **pq) {
    int *a1, *a2;
    Adj *q;
    EMALLOC(1    , &q);
    EMALLOC(md*nv, &q->adj0);
    EMALLOC(md*nv, &q->adj1);
    a1 = q->adj0;
    a2 = q->adj1;
    q->md = md; q->nv = nv;
    ini0(md, nt, nv, faces,      a1, a2);
    *pq = q;
}

void adj_view_ini(const Adj *hst,      Adj_v **pq) {
    int nv, md;
    Adj_v *q;
    if (hst == NULL) ERR("hst == NULL");
    nv = hst->nv; md = hst->md;
    EMALLOC(1, &q);
    Dalloc(&q->adj0, md*nv);
    Dalloc(&q->adj1, md*nv);

    q->nv = nv; q->md = md;
    cH2D(q->adj0, hst->adj0, md*nv);
    cH2D(q->adj1, hst->adj1, md*nv);

    *pq = q;
}
void adj_fin(Adj *q) {
    EFREE(q->adj0);
    EFREE(q->adj1);
    EFREE(q);
}

void adj_view_fin(Adj_v *q) {
    Dfree(q->adj0);
    Dfree(q->adj1);
    EFREE(q);
}
static int fetch(const int *adj, int i) { return adj[i]; }
static int hst0(int md, int nv, int i, const int *adj0, const int *adj1,      AdjMap *m) {
    int i0, i1, i2, i3, i4;
    int j, k;
    if (i >= md * nv)
        ERR("(i = %d) >= (md = %d) * (nv = %d)", i, md, nv);

    i0 = i / md;
    j  = i % md;

    k   = i0 * md;
    i1 = fetch(adj0, k + j);
    if (i1 == -1) return 0;

    i2 = fetch(adj0, k + ((j + 1) % md));
    if (i2 == -1) {
        i2 = fetch(adj0, k    );
        i3 = fetch(adj0, k + 1);
    } else {
        i3 = fetch(adj0, k + ((j + 2) % md));
        if (i3 == -1) i3 = fetch(adj0, k);
    }
    i4 = fetch(adj1, k + j);

    m->i0 = i0; m->i1 = i1; m->i2 = i2; m->i3 = i3; m->i4 = i4;
    return 1;
}

int adj_get_map(int i, const Adj *q,      AdjMap *m) {
    int md, nv;
    const int *adj0, *adj1;
    md = q->md;
    nv = q->nv;
    adj0 = q->adj0;
    adj1 = q->adj1;
    return hst0(md, nv, i, adj0, adj1, m);
}

int adj_get_max(const Adj *q) { return (q->md) * (q->nv); }
static void set(int md, int nv, const Adj *adj,      int *hx, int *hy) {
    AdjMap m;
    int valid, i, i0, i1;
    for (i = 0; i < md*nv; i++) {
        valid = adj_get_map(i, adj,      &m);
        if (!valid) continue;
        i0 = m.i0; i1 = m.i1;
        edg_set(md, i0, i1, i,  hx, hy);
    }
}

static void get(int md, int nv, const Adj *adj, const int *hx, const int *hy,      int *anti) {
    AdjMap m;
    int valid, i, j, i0, i1;
    for (i = 0; i < md*nv; i++) {
        valid = adj_get_map(i, adj,      &m);
        if (!valid) continue;
        i0 = m.i0; i1 = m.i1;

        j = edg_get(md, i1, i0, hx, hy);
        anti[i] = j;
    }
}

static void ini0(int md, int nv, const Adj *adj,      int *anti,       int *hx, int *hy) {
    edg_ini(md, nv,      hx);
    set(md, nv, adj,      hx, hy);
    get(md, nv, adj, hx, hy,      anti);
}

void adj_get_anti(const Adj *adj,      int *anti) {
    int md, nv, n;
    int *hx, *hy;
    md = adj->md;
    nv = adj->nv;
    n = md*nv;
    EMALLOC(n, &hx);
    EMALLOC(n, &hy);
    ini0(md, nv, adj,      anti,       hx, hy);
    EFREE(hx); EFREE(hy);
}





#include <stdio.h>
#include <assert.h>




#ifndef UDX_RBC_RIG_DEF_COMMON
#define UDX_RBC_RIG_DEF_COMMON
#ifndef UDX_RBC_CORE_DEF_COMMON
#define UDX_RBC_CORE_DEF_COMMON
#include "constants.cuh"
#endif
#endif
#ifndef UDX_RBC_RIG_TYPE_COMMON
#define UDX_RBC_RIG_TYPE_COMMON
#ifndef UDX_RBC_CORE_TYPE_COMMON
#define UDX_RBC_CORE_TYPE_COMMON
#include "types.cuh"
#endif
#endif





#define D2D d::MemcpyDeviceToDevice
#define D2H d::MemcpyDeviceToHost
#define H2D d::MemcpyHostToDevice
#define H2H d::MemcpyHostToHost
#define A2A d::MemcpyDefault

#define cD2D(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), D2D))
#define cH2H(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), H2H))
#define cA2A(T, F, n) CC(d::Memcpy((T), (F), (n) * sizeof((F)[0]), A2A))
#define cD2H(H, D, n) CC(d::Memcpy((H), (D), (n) * sizeof((H)[0]), D2H))
#define cH2D(D, H, n) CC(d::Memcpy((D), (H), (n) * sizeof((H)[0]), H2D))

#define aD2D(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), D2D))
#define aH2H(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), H2H))
#define aA2A(T, F, n) CC(d::MemcpyAsync((T), (F), (n) * sizeof((F)[0]), A2A))
#define aD2H(H, D, n) CC(d::MemcpyAsync((H), (D), (n) * sizeof((H)[0]), D2H))
#define aH2D(D, H, n) CC(d::MemcpyAsync((D), (H), (n) * sizeof((H)[0]), H2D))


#define Dfree(D)      CC(d::Free(D))
#define Dalloc(D, n)  CC(d::Malloc((void**)(D), (n) * sizeof(**(D))))


#define Dset(P, v, n) CC(d::Memset(P, v, (n)*sizeof(*(P))))
#define Dzero(P, n)   Dset(P, 0, n)


#define DsetA(P, v, n) CC(d::MemsetAsync(P, v, (n)*sizeof(*(P))))
#define DzeroA(P, n)   DsetA(P, 0, n)


#define _HD_ __host__ __device__

template <typename T1, typename T2>
_HD_ void scal(const T1 a,        T2 *b) {
    b->x *= a;
    b->y *= a;
    b->z *= a;
}

template <typename T1, typename T2>
_HD_ void add(const T1 *a,        T2 *b) {
    b->x += a->x;
    b->y += a->y;
    b->z += a->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void diff(const T1 *a, const T2 *b,      T3 *c) {
    c->x = a->x - b->x;
    c->y = a->y - b->y;
    c->z = a->z - b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ T1 dot(const T2 *a, const T3 *b) {
    return a->x * b->x + a->y * b->y + a->z * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void cross(const T1 *a, const T2 *b,      T3 *c) {
    c->x = a->y * b->z - a->z * b->y;
    c->y = a->z * b->x - a->x * b->z;
    c->z = a->x * b->y - a->y * b->x;
}

template <typename T1, typename T2, typename T3>
_HD_ void scalmult(const T1 *a, const T2 x,      const T3 *b) {
    b->x = a->x * x;
    b->y = a->y * x;
    b->z = a->z * x;
}


template <typename T1, typename T2, typename T3, typename T4>
_HD_ void apxb(const T1 *a, const T2 x, const T3 *b,      T4 *c) {
    c->x = a->x + x * b->x;
    c->y = a->y + x * b->y;
    c->z = a->z + x * b->z;
}

template <typename T1, typename T2, typename T3>
_HD_ void axpy(const T1 a, const T2 *x,        T3 *y) {
    y->x += a * x->x;
    y->y += a * x->y;
    y->z += a * x->z;
}

void msg_ini(int rank);
void msg_print(const char *fmt, ...);
#define ERR(fmt, ...)  do {                                     \
        error_signal(__FILE__, __LINE__, (fmt), ##__VA_ARGS__); \
        error_report();                                         \
        error_abort();                                          \
    } while (0)


#define UC(F) do {                              \
        error_stack_push(__FILE__, __LINE__);   \
        F;                                      \
        if (error_get()) {                      \
            error_report();                     \
            error_abort();                      \
        }                                       \
        error_stack_pop();                      \
    } while (0)


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
void emalloc(size_t,      void**);
void efree(void*);
void *ememcpy(void *dest, const void *src, size_t n);

#define EMALLOC(n, ppdata) UC(emalloc((n)*sizeof(**(ppdata)), (void**)(ppdata)))
#define EFREE(pdata) UC(efree(pdata))
#define EMEMCPY(n, src, dest) ememcpy((void*)(dest), (const void*)(src), (n)*sizeof(*(dest)))

void efopen(const char *fname, const char *mode,      FILE**);
void efclose(FILE*);
void efread(void *ptr, size_t size, size_t nmemb, FILE*);
void efwrite(const void *ptr, size_t size, size_t nmemb, FILE*);
void efgets(char *s, int size, FILE*);

bool same_str(const char *a, const char *b);
const char* get_filename_ext(const char*);



#if   defined(DEV_CUDA)


#elif defined(DEV_CPU)

#ifdef __device__
  #undef  __device__
  #define __device__
#endif

#ifdef __global__
  #undef  __global__
  #define __global__
#endif

#ifdef __host__
  #undef  __host__
  #define __host__
#endif

#ifdef __noinline__
  #undef  __noinline__
  #define __noinline__
#endif

#ifdef __forceinline__
  #undef  __forceinline__
  #define __forceinline__
#endif



#ifdef __constant__
  #undef  __constant__
  #define __constant__
#endif

#ifdef __shared__
  #undef  __shared__
  #define __shared__
#endif

#ifdef __managed__
  #undef  __managed__
  #define __managed__
#endif

#ifdef __restrict__
  #undef  __restrict__
  #define __restrict__
#endif
#else
  #error DEV_* is undefined
#endif
#ifndef UDX_RBC_DEVICE_API_COMMON
#define UDX_RBC_DEVICE_API_COMMON
#ifndef UDX_RBC_RIG_DEVICE_API_COMMON
#define UDX_RBC_RIG_DEVICE_API_COMMON
#ifndef UDX_RBC_CORE_DEVICE_API_COMMON
#define UDX_RBC_CORE_DEVICE_API_COMMON
#include "device.cuh"
#endif
#endif
#endif
#if   defined(DEV_CUDA)
#define  Ifetch(t, i)  tex1Dfetch(t, i)
#define F4fetch(t, i)  tex1Dfetch(t, i)
#define F2fetch(t, i)  tex1Dfetch(t, i)
#define  Ffetch(t, i)  tex1Dfetch(t, i)
#define Tfetch(T, t, i) tex1Dfetch<T>(t, i)

#define Ttex3D(T, to, i, j, k) tex3D<T>(to, i, j, k);
#elif defined(DEV_CPU)
#define atomicAdd(...) (0)

#define atomicExch(...)
#define __float2half_rn(...) (0)
#define __fmaf_rz(...) (0.0)
#define __ldg(p) (*(p))
#define __popc(...)
#define __shfl(...) (0.0)
#define __shfl_down(...) (0.0)
#define __shfl_up(...) (0)
#define __shfl_xor(...) (0)

#define __syncthreads(...)

#define tex3D(...)

#define  Ifetch(t, i)  (0);
#define F4fetch(t, i)  (make_float4(0, 0, 0, 0))
#define F2fetch(t, i)  (make_float2(0, 0))
#define Ffetch(t, i)   (0.0)
#define Tfetch(T, t, i) (T{})

#define Ttex3D(T, to, i, j, k) (T{})
namespace d {
static uint3 threadIdx, blockIdx, gridDim;
static dim3  blockDim;
static int   warpSize = 1;
}

#define threadIdx d::threadIdx
#define blockDim  d::blockDim
#define gridDim   d::gridDim
#define blockIdx  d::blockIdx
#define warpSize  d::warpSize
#else
  #error DEV_* is undefined
#endif


#if   defined(CC_SYNC)

#define CC(ans)                                             \
    do {                                                    \
        d::DeviceSynchronize();                             \
        cc::check((ans), __FILE__, __LINE__);               \
        d::DeviceSynchronize();                             \
    } while (0)
#elif defined(CC_RELEASE)

#define CC(ans)                                             \
    do { cc::check((ans), __FILE__, __LINE__);} while (0)
#else
  #error CC_* is undefined
#endif

namespace cc {
void check(int rc, const char *file, int line);
}




#ifndef UDX_RBC_RIG_KL_COMMON
#define UDX_RBC_RIG_KL_COMMON
#ifndef UDX_RBC_CORE_KL_COMMON
#define UDX_RBC_CORE_KL_COMMON
#include "launch.cuh"
#endif
#endif
#ifndef UDX_RBC_RIG_SHFL_COMMON
#define UDX_RBC_RIG_SHFL_COMMON
#define _I_ static __device__

template <typename T>
_I_ T shfl_down(T val, unsigned int width) {
    return __shfl_down_sync(0xFFFFFFFF, val, width);
}

template <typename T>
_I_ T shfl_up(T val, unsigned int width) {
    return __shfl_up_sync(0xFFFFFFFF, val, width);
}
template <typename T>
_I_ T shfl(T val, unsigned int width) {
    return __shfl_sync(0xFFFFFFFF, val, width);
}

#undef _I_
#endif
#ifndef UDX_RBC_RIG_ALGO_DEV_COMMON
#define UDX_RBC_RIG_ALGO_DEV_COMMON
#define _S_ static __device__
#define _I_ static __device__

template <typename T>
_I_ T warpReduceSum(T v)
{
    for (int offset = warpSize>>1; offset > 0; offset >>= 1)
        v += shfl_down(v, offset);
    return v;
}

_I_ float2 warpReduceSum(float2 val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.x += shfl_down(val.x, offset);
        val.y += shfl_down(val.y, offset);
    }
    return val;
}

_I_ float3 warpReduceSum(float3 val)
{
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val.x += shfl_down(val.x, offset);
        val.y += shfl_down(val.y, offset);
        val.z += shfl_down(val.z, offset);
    }
    return val;
}

#undef _S_
#undef _I_

#endif

struct RbcCom;
struct float3;
struct Particle;

void rbc_com_ini(int nv, int nm_max,      RbcCom**);
void rbc_com_fin(     RbcCom*);
void rbc_com_apply(RbcCom*, int nm, const Particle*,      float3 **rr, float3 **vv);

struct RbcCom {
    int nv;
    int max_cell;
    float3 *drr, *dvv;
    float3 *hrr, *hvv;
};

__global__ void rbccom_reduce_props_kernel(int nv, const Particle *pp,      float3 *rr, float3 *vv) {
    int mid, i;
    float3 r, v;
    i   = threadIdx.x + blockIdx.x * blockDim.x;
    mid = blockIdx.y;

    r = v = make_float3(0, 0, 0);

    if (i < nv) {
        enum {X, Y, Z};
        const float *r0 = pp[mid * nv + i].r;
        const float *v0 = pp[mid * nv + i].v;
        r = make_float3(r0[X], r0[Y], r0[Z]);
        v = make_float3(v0[X], v0[Y], v0[Z]);
    }

    r = warpReduceSum(r);
    v = warpReduceSum(v);

    if ((threadIdx.x % warpSize) == 0) {
        atomicAdd(&rr[mid].x, r.x);
        atomicAdd(&rr[mid].y, r.y);
        atomicAdd(&rr[mid].z, r.z);

        atomicAdd(&vv[mid].x, v.x);
        atomicAdd(&vv[mid].y, v.y);
        atomicAdd(&vv[mid].z, v.z);
    }
}void rbc_com_ini(int nv, int max_cell,      RbcCom **pq) {
    RbcCom *q;
    size_t sz;
    EMALLOC(1, &q);

    sz = max_cell * sizeof(float3);
    CC(d::alloc_pinned((void**) &q->hrr, sz));
    CC(d::alloc_pinned((void**) &q->hvv, sz));
    Dalloc(&q->drr, max_cell);
    Dalloc(&q->dvv, max_cell);
    q->nv = nv; q->max_cell = max_cell;
    *pq = q;
}
void rbc_com_fin(     RbcCom *q) {
    CC(d::FreeHost(q->hrr));
    CC(d::FreeHost(q->hvv));
    Dfree(q->drr);
    Dfree(q->dvv);
    EFREE(q);
}
static void reini(int nm,      float3 *h) {
    if (nm) CC(d::MemsetAsync(h, 0, nm * sizeof(float3)));
}

static void reduce(int nm, int nv, const Particle *pp,      float3 *rr, float3 *vv) {
    dim3 thrd(128, 1);
    dim3 blck(ceiln(nv, thrd.x), nm);

    KL(rbccom_reduce_props_kernel, (blck, thrd), (nv, pp,      rr, vv));
}

static void download(int nm, const float3 *d,      float3 *h) {
    if (nm) CC(d::MemcpyAsync(h, d, nm * sizeof(float3), D2H));
}

static void normalize(int nm, int nv,      float3 *h) {
    float fac = 1.f / nv;
    for (int i = 0; i < nm; ++i)
        scal(fac, h + i);
}

static void compute(int nm, const Particle *pp,      RbcCom *q) {
    int nv;
    nv = q->nv;

    reini(nm,      q->drr);
    reini(nm,      q->dvv);

    reduce(nm, nv, pp,      q->drr, q->dvv);

    download(nm, q->drr,      q->hrr);
    download(nm, q->dvv,      q->hvv);
    dSync();

    normalize(nm, nv,      q->hrr);
    normalize(nm, nv,      q->hvv);
}

void rbc_com_apply(RbcCom *q, int nm, const Particle *pp,      float3 **prr, float3 **pvv) {
    int max_cell;
    max_cell = q->max_cell;
    if (nm > max_cell)
        ERR("nm=%d > max_cell=%d", nm, max_cell);
    if (!d::is_device_pointer(pp))
        ERR("`pp` is not a device pointer");
    compute(nm, pp,      q);
    *prr = q->hrr; *pvv = q->hvv;
}
