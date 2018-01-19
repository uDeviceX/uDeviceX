struct DRigPack;
struct DRigComm;
struct DRigUnpack;

// tag::interface[]
void drig_pack_ini(int maxns, int nv, DRigPack **p);
void drig_comm_ini(MPI_Comm comm, /**/ DRigComm **c);
void drig_unpack_ini(int maxns, int nv, DRigUnpack **u);

void drig_pack_fin(DRigPack *p);
void drig_comm_fin(DRigComm *c);
void drig_unpack_fin(DRigUnpack *u);

void drig_build_map(int ns, const Solid *ss, /**/ DRigPack *p);
void drig_pack(int ns, int nv, const Solid *ss, const Particle *ipp, /**/ DRigPack *p);
void drig_download(DRigPack *p);

void drig_post_recv(DRigComm *c, DRigUnpack *u);
void drig_post_send(DRigPack *p, DRigComm *c);
void drig_wait_recv(DRigComm *c, DRigUnpack *u);
void drig_wait_send(DRigComm *c);

void drig_unpack_bulk(const DRigPack *p, /**/ RigQuants *q);
void drig_unpack_halo(const DRigUnpack *u, /**/ RigQuants *q);
// end::interface[]
