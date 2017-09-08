#include <mpi.h>
#include <vector>
#include <assert.h>

#include "conf.h"
#include "inc/conf.h"

#include "inc/type.h"
#include "mpi/wrapper.h"
#include "mpi/glb.h"

namespace wall { namespace sub {
void exch(/*io*/ Particle *pp, int *n) { /* exchange pp(hst) between processors */
  #define isize(v) ((int)(v).size()) /* [i]nteger [size] */
  assert(sizeof(Particle) == 6 * sizeof(float)); /* :TODO: dependencies */
  enum {X, Y, Z};
  int i, j, c;
  int dstranks[26], remsizes[26], recv_tags[26];
  for (i = 0; i < 26; ++i) {
    int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
    recv_tags[i] =
      (2 - d[X]) % 3 + 3 * ((2 - d[Y]) % 3 + 3 * ((2 - d[Z]) % 3));
    int co_ne[3], ranks[3] = {m::coords[X], m::coords[Y], m::coords[Z]};
    for (c = 0; c < 3; ++c) co_ne[c] = ranks[c] + d[c];
    m::Cart_rank(m::cart, co_ne, dstranks + i);
  }

  // send local counts - receive remote counts
  {
    for (i = 0; i < 26; ++i) remsizes[i] = -1;
    MPI_Request reqrecv[26], reqsend[26];
    MPI_Status  statuses[26];
    for (i = 0; i < 26; ++i)
      m::Irecv(remsizes + i, 1, MPI_INTEGER, dstranks[i],
                  123 + recv_tags[i], m::cart, reqrecv + i);
    for (i = 0; i < 26; ++i)
        m::Isend(n, 1, MPI_INTEGER, dstranks[i], 123 + i, m::cart, reqsend + i);
    m::Waitall(26, reqrecv, statuses);
    m::Waitall(26, reqsend, statuses);
  }

  std::vector<Particle> remote[26];
  // send local data - receive remote data
  {
    for (i = 0; i < 26; ++i) remote[i].resize(remsizes[i]);
    MPI_Request reqrecv[26], reqsend[26];
    MPI_Status  statuses[26];
    for (i = 0; i < 26; ++i)
      m::Irecv(remote[i].data(), isize(remote[i]) * 6, MPI_FLOAT,
                  dstranks[i], 321 + recv_tags[i], m::cart,
                  reqrecv + i);
    for (i = 0; i < 26; ++i)
      m::Isend(pp, (*n) * 6, MPI_FLOAT,
                  dstranks[i], 321 + i, m::cart, reqsend + i);
    m::Waitall(26, reqrecv, statuses);
    m::Waitall(26, reqsend, statuses);
  }
  m::Barrier(m::cart);

  int L[3] = {XS, YS, ZS}, WM[3] = {XWM, YWM, ZWM};
  float lo[3], hi[3];
  for (c = 0; c < 3; c ++) {
    lo[c] = -0.5*L[c] - WM[c];
    hi[c] =  0.5*L[c] + WM[c];
  }

  for (i = 0; i < 26; ++i) {
    int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};
    for (j = 0; j < isize(remote[i]); ++j) {
      Particle p = remote[i][j];
      for (c = 0; c < 3; ++c) {
	p.r[c] += d[c] * L[c];
	if (p.r[c] < lo[c] || p.r[c] >= hi[c]) goto next;
      }
      pp[(*n)++] = p;
    next: ;
    }
  }
}

}} /* namespace */
