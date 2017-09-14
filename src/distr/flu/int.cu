#include <mpi.h>

#include "inc/type.h"
#include "mpi/basetags.h"
#include "comm/imp.h"

#include "map.h"
#include "imp.h"

#include "int.h"

namespace distr {
namespace flu {

/* map */

void build_map(int n, const Particle *pp, Pack *p) {
    build_map(n, pp, p->map);
}

/* pack */

void pack_pp(const Particle *pp, int n, /**/ Pack *p) {
    pack_pp(p->map, pp, n, /**/ p->dpp);
}

void pack_ii(const int *ii, int n, /**/ Pack *p) {
    pack_ii(p->map, ii, n, /**/ p->dii);
}

void pack_cc(const int *cc, int n, /**/ Pack *p) {
    pack_ii(p->map, cc, n, /**/ p->dcc);
}

/* communication */
void post_recv(Comm *c, Unpack *u);
void post_send(Pack *p, Comm *c);
void wait_recv(Comm *c, Unpack *u);
void wait_send(Comm *s);

/* unpack */
void unpack_pp(/**/ Unpack *u);
void unpack_ii(/**/ Unpack *u);
void unpack_cc(/**/ Unpack *u);

/* cell lists */


} // flu
} // distr
