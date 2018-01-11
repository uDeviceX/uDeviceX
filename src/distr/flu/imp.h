struct DFluPack;
struct DFluComm;
struct DFluUnpack;

// tag::interface[]
void dflu_pack_ini(int maxdensity, DFluPack **p);
void dflu_comm_ini(MPI_Comm comm, /**/ DFluComm **c);
void dflu_unpack_ini(int maxdensity, DFluUnpack **u);

void dflu_pack_fin(DFluPack *p);
void dflu_comm_fin(DFluComm *c);
void dflu_unpack_fin(DFluUnpack *u);

/* map */
void dflu_build_map(int n, const PartList lp, DFluPack *p);

using namespace flu;

/* pack */
void dflu_pack(const Quants *q, /**/ DFluPack *p);

void dflu_download(DFluPack *p);

/* communication */
void dflu_post_recv(DFluComm *c, DFluUnpack *u);
void dflu_post_send(DFluPack *p, DFluComm   *c);
void dflu_wait_recv(DFluComm *c, DFluUnpack *u);
void dflu_wait_send(DFluComm *c);

/* unpack */
void dflu_unpack(/**/ DFluUnpack *u);
// end::interface[]


// tag::clist[]
/* cell lists */
void dflu_bulk(PartList lp, /**/ Quants *q);
void dflu_halo(const DFluUnpack *u, /**/ Quants *q);
void dflu_gather(int ndead, const DFluPack *p, const DFluUnpack *u, /**/ Quants *q);
// end::clist[]
