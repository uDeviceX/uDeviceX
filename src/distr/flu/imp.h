struct DFluPack;
struct DFluComm;
struct DFluUnpack;
struct DFluStatus;

// tag::interface[]
void dflu_pack_ini(int maxdensity, DFluPack **p);
void dflu_comm_ini(MPI_Comm comm, /**/ DFluComm **c);
void dflu_unpack_ini(int maxdensity, DFluUnpack **u);

void dflu_pack_fin(DFluPack *p);
void dflu_comm_fin(DFluComm *c);
void dflu_unpack_fin(DFluUnpack *u);

/* map */
void dflu_build_map(int n, const PartList lp, DFluPack *p);

/* pack */
void dflu_pack(const FluQuants *q, /**/ DFluPack *p);

void dflu_download(DFluPack *p);

/* communication */
void dflu_post_recv(DFluComm *c, DFluUnpack *u);
void dflu_post_send(DFluPack *p, DFluComm   *c);
void dflu_wait_recv(DFluComm *c, DFluUnpack *u);
void dflu_wait_send(DFluComm *c);

/* status */
void dflu_status_ini(DFluStatus **s);
void dflu_status_fin(DFluStatus  *s);
int  dflu_status_success(DFluStatus *s);
void dflu_status_log(DFluStatus *s);

/* unpack */
void dflu_unpack(/**/ DFluUnpack *u);
// end::interface[]


// tag::clist[]
/* cell lists */
void dflu_bulk(PartList lp, /**/ FluQuants *q);
void dflu_halo(const DFluUnpack *u, /**/ FluQuants *q);
void dflu_gather(int ndead, const DFluPack *p, const DFluUnpack *u, /**/ FluQuants *q);
// end::clist[]
