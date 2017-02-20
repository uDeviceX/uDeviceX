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

class RedistributeParticles {
public:
  void pack(Particle * p, int n, cudaStream_t stream);
  void send();
  void bulk(int nparticles, int * cellstarts, int * cellcounts, cudaStream_t mystream);
  int recv_count(cudaStream_t);
  void recv_unpack(Particle * particles, float4 * xyzouvwo, ushort4 * xyzo_half, int nparticles,
                   int * cellstarts, int * cellcounts, cudaStream_t);
  RedistributeParticles(MPI_Comm cartcomm);
  void adjust_message_sizes(ExpectedMessageSizes sizes);
  ~RedistributeParticles();
  

  void _post_recv();
  void _cancel_recv();
  void _adjust_send_buffers(int capacities[27]);
  bool _adjust_recv_buffers(int capacities[27]);

};

