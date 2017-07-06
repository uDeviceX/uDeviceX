// see the vanilla version of this code for details about how this class
// operates
namespace dpd {
int ncells;                 /* total number of cells in halo           */
MPI_Request sendreq[26 * 2], recvreq[26], sendcellsreq[26], recvcellsreq[26],
    sendcountreq[26], recvcountreq[26];
int recv_tags[26], recv_counts[26];
int dstranks[26];

/* refugees from halo.h */
typedef Sarray<int,  26> int26;
typedef Sarray<int,  27> int27;
typedef Sarray<int*, 26> intp26;
typedef Sarray<Particle*, 26> Particlep26;

int27 fragstarts;           /* cumulative sum of number of cells for each fragment */
intp26 srccells, dstcells;

/* fragments of halo */
namespace frag {
intp26 str, cnt, cum, ii;   /* see /doc/remote.md                        */
int26 nc, capacity;         /* number of cells per fragment              */
Particlep26 pp;             /* buffer of particles for each fragment     */
int *np, *nphst;  /* number of particles on the device and host (pinned) */
}
}
