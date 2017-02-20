class ComputeContact {
  int nsolutes;

  SimpleDeviceBuffer<uchar4> subindices;
  SimpleDeviceBuffer<unsigned char> compressed_cellscount;
  SimpleDeviceBuffer<int> cellsentries, cellsstart, cellscount;

  Logistic::KISS local_trunk;

 public:

  ComputeContact(MPI_Comm comm);

  void build_cells(std::vector<ParticlesWrap> wsolutes, cudaStream_t stream);

  void bulk(std::vector<ParticlesWrap> wsolutes, cudaStream_t stream);

  /*override of SoluteExchange::Visitor::halo*/
  void halo(ParticlesWrap solutes[26], cudaStream_t stream);
};
