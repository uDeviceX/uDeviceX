struct DRbcPack;
struct DRbcComm;
struct DRbcUnpack;

// tag::interface[]
void drbc_pack_ini(int maxc, int nv, DRbcPack **p);
void drbc_comm_ini(MPI_Comm comm, /**/ DRbcComm **c);
void drbc_unpack_ini(int maxc, int nv, DRbcUnpack **u);

void drbc_pack_fin(DRbcPack *p);
void drbc_comm_fin(DRbcComm *c);
void drbc_unpack_fin(DRbcUnpack *u);

using namespace rbc;

void drbc_build_map(int nc, int nv, const Particle *pp, DRbcPack *p);
void drbc_pack(const rbc::Quants *q, /**/ DRbcPack *p);
void drbc_download(DRbcPack *p);

void drbc_post_recv(DRbcComm *c, DRbcUnpack *u);
void drbc_post_send(DRbcPack *p, DRbcComm *c);
void drbc_wait_recv(DRbcComm *c, DRbcUnpack *u);
void drbc_wait_send(DRbcComm *c);

void drbc_unpack_bulk(const DRbcPack *p, /**/ rbc::Quants *q);
void drbc_unpack_halo(const DRbcUnpack *u, /**/ rbc::Quants *q);
// end::interface[]
