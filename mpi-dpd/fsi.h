class ComputeFSI {
  SolventWrap wsolvent;
  Logistic::KISS local_trunk;
public:
  void bind_solvent(SolventWrap wrap) {wsolvent = wrap;}
  ComputeFSI(MPI_Comm comm);
  void bulk(std::vector<ParticlesWrap> wsolutes, cudaStream_t stream);
  /*override of SoluteExchange::Visitor::halo*/
  void halo(ParticlesWrap solutes[26], cudaStream_t stream);
};
