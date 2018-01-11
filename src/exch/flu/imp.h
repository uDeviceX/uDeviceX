namespace exch {
namespace flu {

void eflu_pack_ini(int maxd, Pack *p);
void eflu_comm_ini(MPI_Comm comm, /**/ Comm *c);
void eflu_unpack_ini(int maxd, Unpack *u);

void eflu_pack_fin(Pack *p);
void eflu_comm_fin(Comm *c);
void eflu_unpack_fin(Unpack *u);

void eflu_compute_map(const int *start, const int *count, /**/ Pack *p);
void eflu_download_cell_starts(/**/ Pack *p);

void eflu_pack(const Cloud *cloud, /**/ Pack *p);
void eflu_download_data(Pack *p);

void eflu_post_recv(Comm *c, Unpack *u);
void eflu_post_send(Pack *p, Comm *c);
void eflu_wait_recv(Comm *c, Unpack *u);
void eflu_wait_send(Comm *c);

void eflu_unpack(Unpack *u);

using ::flu::LFrag26;
using ::flu::RFrag26;

void eflu_get_local_frags(const Pack *p, /**/ LFrag26 *lfrags);
void eflu_get_remote_frags(const Unpack *u, /**/ RFrag26 *rfrags);
    
} // flu
} // exch
