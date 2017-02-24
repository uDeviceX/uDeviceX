/*** from redistribute-particles.h ***/
namespace sdstr {
  int basetag = 950;
  MPI_Comm cartcomm_rdst;
  float safety_factor = 1.2;
  int neighbor_ranks[27], recv_tags[27],
    default_message_sizes[27], send_sizes[27], recv_sizes[27],
    nsendmsgreq, nexpected, nbulk, nhalo, nhalo_padded, myrank;
  bool firstcall;
  int nactiveneighbors;
  MPI_Request sendcountreq[27], recvcountreq[27],
    sendmsgreq[27 * 2], recvmsgreq[27 * 2];
  cudaEvent_t evpacking, evsizes;

  float * pinnedhost_sendbufs[27], * pinnedhost_recvbufs[27];
  struct UnpackBuffer {
    float2 * buffer;
    int capacity;
  };

  struct PackBuffer {
    float2 * buffer;
    int capacity;
    int * scattered_indices;
  };

  PackBuffer packbuffers[27];
  UnpackBuffer unpackbuffers[27];

  PinnedHostBuffer<bool> *failure;
  PinnedHostBuffer<int> *packsizes;
  DeviceBuffer<unsigned char> *compressed_cellcounts;
  DeviceBuffer<Particle> *remote_particles;
  
  DeviceBuffer<uchar4>   *subindices_remote;
  DeviceBuffer<uchar4>   *subindices;
  DeviceBuffer<uint>     *scattered_indices;

}
