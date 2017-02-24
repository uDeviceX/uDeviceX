namespace sim {
  void sim_init(MPI_Comm cartcomm_, MPI_Comm activecomm_);
  void sim_run();
  void sim_close();
}
