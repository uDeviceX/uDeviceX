namespace exch {
namespace flu {

void eflu_pack_ini(int maxd, EFluPack *p);
void eflu_comm_ini(MPI_Comm comm, /**/ EFluComm *c);
void eflu_unpack_ini(int maxd, EFluUnpack *u);

void eflu_pack_fin(EFluPack *p);
void eflu_comm_fin(EFluComm *c);
void eflu_unpack_fin(EFluUnpack *u);

void eflu_compute_map(const int *start, const int *count, /**/ EFluPack *p);
void eflu_download_cell_starts(/**/ EFluPack *p);

void eflu_pack(const Cloud *cloud, /**/ EFluPack *p);
void eflu_download_data(EFluPack *p);

void eflu_post_recv(EFluComm *c, EFluUnpack *u);
void eflu_post_send(EFluPack *p, EFluComm *c);
void eflu_wait_recv(EFluComm *c, EFluUnpack *u);
void eflu_wait_send(EFluComm *c);

void eflu_unpack(EFluUnpack *u);

using ::flu::LFrag26;
using ::flu::RFrag26;

void eflu_get_local_frags(const EFluPack *p, /**/ LFrag26 *lfrags);
void eflu_get_remote_frags(const EFluUnpack *u, /**/ RFrag26 *rfrags);
    
} // flu
} // exch
