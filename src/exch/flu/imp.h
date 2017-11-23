namespace exch {
namespace flu {

void ini(int maxd, Pack *p);
void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c);
void ini(int maxd, Unpack *u);

void fin(Pack *p);
void fin(Comm *c);
void fin(Unpack *u);


void compute_map(const int *start, const int *count, /**/ Pack *p);
void download_cell_starts(/**/ Pack *p);

void pack(const Cloud *cloud, /**/ Pack *p);
void download_data(Pack *p);

void post_recv(Comm *c, Unpack *u);
void post_send(Pack *p, Comm *c);
void wait_recv(Comm *c, Unpack *u);
void wait_send(Comm *c);

void unpack(Unpack *u);

using ::flu::LFrag26;
using ::flu::RFrag26;

void get_local_frags(const Pack *p, LFrag26 *lfrags);
void get_remote_frags(const Unpack *u, /**/ RFrag26 *rfrags);
    
} // flu
} // exch
