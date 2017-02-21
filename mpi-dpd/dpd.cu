#include <utility>
#include <cell-lists.h>
#include <cuda-dpd.h>
#include <dpd-rng.h>
#include <mpi.h>
#include ".conf.h" /* configuration file (copy from .conf.test.h) */
#include "common.h"
#include "solvent-exchange.h"
#include "dpd.h"
#include "dpd-forces.h"
#include "last_bit_float.h"

#include "bipsbatch.decl.h"
#include "bipsbatch.impl.h"

ComputeDPD::ComputeDPD(MPI_Comm cartcomm)
    : SolventExchange(cartcomm, 0) {
  local_trunk = new Logistic::KISS(0, 0, 0, 0);
  init1(cartcomm);
}

ComputeDPD::~ComputeDPD() {
  for (int i = 1; i < 26; i++) delete interrank_trunks[i];
  delete local_trunk;
}

void ComputeDPD::init1(MPI_Comm cartcomm) {
    int myrank;
  MC(MPI_Comm_rank(cartcomm, &myrank));

  for (int i = 0; i < 26; ++i) {
    int d[3] = {(i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1};

    int coordsneighbor[3];
    for (int c = 0; c < 3; ++c)
      coordsneighbor[c] = (coords[c] + d[c] + dims[c]) % dims[c];

    int indx[3];
    for (int c = 0; c < 3; ++c)
      indx[c] = min(coords[c], coordsneighbor[c]) * dims[c] +
                max(coords[c], coordsneighbor[c]);

    int interrank_seed_base =
        indx[0] + dims[0] * dims[0] * (indx[1] + dims[1] * dims[1] * indx[2]);

    int interrank_seed_offset;

    {
      bool isplus =
          d[0] + d[1] + d[2] > 0 ||
          d[0] + d[1] + d[2] == 0 &&
              (d[0] > 0 || d[0] == 0 && (d[1] > 0 || d[1] == 0 && d[2] > 0));

      int mysign = 2 * isplus - 1;

      int v[3] = {1 + mysign * d[0], 1 + mysign * d[1], 1 + mysign * d[2]};

      interrank_seed_offset = v[0] + 3 * (v[1] + 3 * v[2]);
    }

    int interrank_seed = interrank_seed_base + interrank_seed_offset;

    interrank_trunks[i] = new Logistic::KISS(390 + interrank_seed,
					     interrank_seed + 615, 12309, 23094);

    int dstrank = dstranks[i];

    if (dstrank != myrank)
      interrank_masks[i] = min(dstrank, myrank) == myrank;
    else {
      int alter_ego =
          (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));
      interrank_masks[i] = min(i, alter_ego) == i;
    }
  }
}

void ComputeDPD::local_interactions(Particle *xyzuvw, float4 *xyzouvwo,
                                    ushort4 *xyzo_half, int n, Acceleration *a,
                                    int *cellsstart, int *cellscount,
                                    cudaStream_t stream) {
  if (n > 0) forces_dpd_cuda_nohost(
      (float *)xyzuvw, xyzouvwo, xyzo_half, (float *)a, n, cellsstart,
      cellscount, 1, XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN,
      1. / sqrt(dt), local_trunk->get_float(), stream);
}

void ComputeDPD::remote_interactions(Particle *p, int n, Acceleration *a,
                                     cudaStream_t stream,
                                     cudaStream_t uploadstream) {
  CC(cudaPeekAtLastError());

  static BipsBatch::BatchInfo infos[26];

  for (int i = 0; i < 26; ++i) {
    int dx = (i + 2) % 3 - 1;
    int dy = (i / 3 + 2) % 3 - 1;
    int dz = (i / 9 + 2) % 3 - 1;

    int m0 = 0 == dx;
    int m1 = 0 == dy;
    int m2 = 0 == dz;

    BipsBatch::BatchInfo entry = {
        (float *)sendhalos[i].dbuf.D,
        (float2 *)recvhalos[i].dbuf.D,
        interrank_trunks[i]->get_float(),
        sendhalos[i].dbuf.S,
        recvhalos[i].dbuf.S,
        interrank_masks[i],
        recvhalos[i].dcellstarts.D,
        sendhalos[i].scattered_entries.D,
        dx,
        dy,
        dz,
        1 + m0 * (XSIZE_SUBDOMAIN - 1),
        1 + m1 * (YSIZE_SUBDOMAIN - 1),
        1 + m2 * (ZSIZE_SUBDOMAIN - 1),
        (BipsBatch::HaloType)(abs(dx) + abs(dy) + abs(dz))};

    infos[i] = entry;
  }

  BipsBatch::interactions(1. / sqrt(dt), infos, stream, uploadstream,
                          (float *)a, n);

  CC(cudaPeekAtLastError());
}
