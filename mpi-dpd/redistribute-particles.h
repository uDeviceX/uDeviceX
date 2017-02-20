/*
 *  redistribute-particles.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-14.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

extern int basetag;
extern float safety_factor;
extern MPI_Comm cartcomm;

class RedistributeParticles {
public:
  struct UnpackBuffer {
    float2 * buffer;
    int capacity;
  };

  struct PackBuffer {
    float2 * buffer;
    int capacity;
    int * scattered_indices;
  };

  void pack(Particle * p, int n, cudaStream_t stream);
  void send();
  void bulk(int nparticles, int * cellstarts, int * cellcounts, cudaStream_t mystream);
  int recv_count(cudaStream_t);
  void recv_unpack(Particle * particles, float4 * xyzouvwo, ushort4 * xyzo_half, int nparticles,
                   int * cellstarts, int * cellcounts, cudaStream_t);
  RedistributeParticles(MPI_Comm cartcomm);
  void adjust_message_sizes(ExpectedMessageSizes sizes);
  ~RedistributeParticles();
  int pack_size(int code) { return send_sizes[code]; }
  float pinned_data(int code, int entry) {return pinnedhost_sendbufs[code][entry]; }

  bool firstcall;
  int neighbor_ranks[27], recv_tags[27],
    default_message_sizes[27], send_sizes[27], recv_sizes[27],
    nsendmsgreq, nexpected, nbulk, nhalo, nhalo_padded, myrank;
  int nactiveneighbors;
  MPI_Request sendcountreq[27], recvcountreq[27], sendmsgreq[27 * 2], recvmsgreq[27 * 2];
  cudaEvent_t evpacking, evsizes;
  
  PinnedHostBuffer<bool> failure;
  PinnedHostBuffer<int> packsizes;
  float * pinnedhost_sendbufs[27], * pinnedhost_recvbufs[27];
  PackBuffer packbuffers[27];
  UnpackBuffer unpackbuffers[27];
  SimpleDeviceBuffer<unsigned char> compressed_cellcounts;
  SimpleDeviceBuffer<Particle> remote_particles;
  SimpleDeviceBuffer<uint> scattered_indices;
  SimpleDeviceBuffer<uchar4> subindices, subindices_remote;

  void _waitall(MPI_Request * reqs, int n) {
    MPI_Status statuses[n];
    MPI_CHECK( MPI_Waitall(n, reqs, statuses) );
  }

  void _post_recv();
  void _cancel_recv();
  void _adjust_send_buffers(int capacities[27]);
  bool _adjust_recv_buffers(int capacities[27]);

};

