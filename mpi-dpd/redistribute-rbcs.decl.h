namespace RedistRBC {
  SimpleDeviceBuffer<Particle> bulk;
  PinnedHostBuffer<Particle> halo_recvbufs[27], halo_sendbufs[27];
  PinnedHostBuffer<float3> minextents, maxextents;

  MPI_Comm cartcomm;
  MPI_Request sendcountreq[26];
  std::vector<MPI_Request> sendreq, recvreq, recvcountreq;
  int myrank, periods[3], coords[3], rankneighbors[27],
      anti_rankneighbors[27];
  int recv_counts[27];

  int nvertices, arriving, notleaving;
  cudaEvent_t evextents;
}
