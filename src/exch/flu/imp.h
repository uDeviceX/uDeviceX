namespace exch {
namespace flu {

void compute_map(const int *start, const int *count, /**/ Pack *p);
void download_cell_starts(/**/ Pack *p);

void pack(const Cloud *cloud, /**/ Pack *p);
void download_data(Pack *p);

void post_recv(Comm *c, Unpack *u);
void post_send(Pack *p, Comm *c);
void wait_recv(Comm *c, Unpack *u);
void wait_send(Comm *c);

void unpack(Unpack *u);

} // flu
} // exch
