namespace distr {
namespace flu {

using namespace comm;

struct Pack {
    Map map;
    dBags dpp, dii, dcc;
    hBags hpp, hii, hcc;
    int nbulk;
};

struct Comm {
    Stamp pp, ii, cc;
};

struct Unpack {
    hBags hpp, hii, hcc;
    Particle *ppre;
    int *iire, *ccre;
    int nhalo;
};

void ini(float maxdensity, Pack *p);
void ini(MPI_Comm comm, /*io*/ basetags::TagGen *tg, /**/ Comm *c);
void ini(float maxdensity, Unpack *u);

void fin(Pack *p);
void fin(Comm *c);
void fin(Unpack *u);


/* map */
void build_map(int n, const Particle *pp, Pack *p);

/* pack */
void pack_pp(const Particle *pp, int n, /**/ Pack *p);
void pack_ii(const int *ii, int n, /**/ Pack *p);
void pack_cc(const int *cc, int n, /**/ Pack *p);

void download(Pack *p);

/* communication */
void post_recv(Comm *c, Unpack *u);
void post_send(Pack *p, Comm *c);
void wait_recv(Comm *c, Unpack *u);
void wait_send(Comm *c);

/* unpack */
void unpack_pp(/**/ Unpack *u);
void unpack_ii(/**/ Unpack *u);
void unpack_cc(/**/ Unpack *u);

/* cell lists */


} // flu
} // distr
