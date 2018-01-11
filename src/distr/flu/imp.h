namespace distr {
namespace flu {

// tag::interface[]
void dflu_pack_ini(int maxdensity, Pack *p);
void dflu_comm_ini(MPI_Comm comm, /**/ Comm *c);
void dflu_unpack_ini(int maxdensity, Unpack *u);

void dflu_pack_fin(Pack *p);
void dflu_comm_fin(Comm *c);
void dflu_unpack_fin(Unpack *u);

/* map */
void dflu_build_map(int n, const PartList lp, Pack *p);

using namespace ::flu;

/* pack */
void dflu_pack(const Quants *q, /**/ Pack *p);

void dflu_download(Pack *p);

/* communication */
void dflu_post_recv(Comm *c, Unpack *u);
void dflu_post_send(Pack *p, Comm   *c);
void dflu_wait_recv(Comm *c, Unpack *u);
void dflu_wait_send(Comm *c);

/* unpack */
void dflu_unpack(/**/ Unpack *u);
// end::interface[]


// tag::clist[]
/* cell lists */
void dflu_bulk(PartList lp, /**/ Quants *q);
void dflu_halo(const Unpack *u, /**/ Quants *q);
void dflu_gather(int ndead, const Pack *p, const Unpack *u, /**/ Quants *q);
// end::clist[]

} // flu
} // distr
