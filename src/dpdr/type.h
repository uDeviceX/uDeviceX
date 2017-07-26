namespace dpdr {
namespace sub {
typedef Sarray<int,  26> int26;
typedef Sarray<int,  27> int27;
typedef Sarray<int*, 26> intp26;
typedef Sarray<Particle*, 26> Particlep26;

/* Structure containing all kinds of requests a fragment can have */
struct Reqs {
    MPI_Request pp[26], cells[26], counts[26];
};

struct Bbufs { /* basic buffer : common to Send and Recv */
    intp26 cum;                /* cellstarts for each fragment (frag coords)      */
    Particlep26 pp;            /* buffer of particles for each fragment           */

    /* pinned buffers */
    Particlep26 ppdev, pphst;  /* pinned memory for transfering particles         */
    intp26 cumdev, cumhst;     /* pinned memory for transfering local cum sum     */
};

struct Sbufs : Bbufs {
    intp26 str, cnt;           /* cell starts and counts for each fragment (bulk coords) */
    intp26 ii;                 /* scattered indices                                      */
};

struct Rbufs : Bbufs {};

struct Ibuf {
    intp26 ii;                  /* int data on device for each fragment    */

    /* pinned buffers */
    intp26 iihst, iidev;        /* pinned memory for transfering int data  */
};

struct SIbuf : Ibuf {};
struct RIbuf : Ibuf {};
}
}
