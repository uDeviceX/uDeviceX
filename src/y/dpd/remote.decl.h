// see the vanilla version of this code for details about how this class
// operates
namespace dpd {
int ncells;
MPI_Request sendreq[26 * 2], recvreq[26], sendcellsreq[26], recvcellsreq[26],
    sendcountreq[26], recvcountreq[26];
int recv_tags[26], recv_counts[26];
int dstranks[26];

/* refugees from halo.h */
typedef Sarray<int,  26> int26;
typedef Sarray<int,  27> int27;
typedef Sarray<int*, 26> intp26;
typedef Sarray<Particle*, 26> Particlep26;

int27 cellpackstarts;
intp26 srccells, dstcells;

/* fragments of halo */
namespace frag {
intp26 start, count, scan, indices;
int26 size, capacity;
Particlep26 pp;
}

// zero-copy allocation for acquiring the message offsets in the gpu send
// buffer
int *required_send_bag_size, *required_send_bag_size_host;

// plain copy of the offsets for the cpu (i speculate that reading multiple
// times the zero-copy entries is slower)
int nsendreq;
cudaEvent_t evfillall, evuploaded, evdownloaded;
}
