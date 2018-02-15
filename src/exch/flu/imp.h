struct EFluPack;
struct EFluComm;
struct EFluUnpack;
struct PaArray;

// tag::mem[]
void eflu_pack_ini(bool colors, int3 L, int maxd, EFluPack **p);
void eflu_comm_ini(bool colors, MPI_Comm comm, /**/ EFluComm **c);
void eflu_unpack_ini(bool colors, int3 L, int maxd, EFluUnpack **u);

void eflu_pack_fin(EFluPack *p);
void eflu_comm_fin(EFluComm *c);
void eflu_unpack_fin(EFluUnpack *u);
// end::mem[]

// tag::map[]
void eflu_compute_map(const int *start, const int *count, /**/ EFluPack *p);
void eflu_download_cell_starts(/**/ EFluPack *p);
// end::map[]

// tag::pack[]
void eflu_pack(const PaArray *parray, /**/ EFluPack *p);
void eflu_download_data(EFluPack *p);
// end::pack[]

// tag::com[]
void eflu_post_recv(EFluComm *c, EFluUnpack *u);
void eflu_post_send(EFluPack *p, EFluComm *c);
void eflu_wait_recv(EFluComm *c, EFluUnpack *u);
void eflu_wait_send(EFluComm *c);
// tag::end[]

// tag::unpack[]
void eflu_unpack(EFluUnpack *u);

using flu::LFrag26;
using flu::RFrag26;

void eflu_get_local_frags(const EFluPack *p, /**/ LFrag26 *lfrags);
void eflu_get_remote_frags(const EFluUnpack *u, /**/ RFrag26 *rfrags);
// end::unpack[]

