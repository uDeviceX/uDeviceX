void ply_dump(const char * filename,
              int *mesh_indices, const int ninstances, const int ntriangles_per_instance, Particle * _particles,
              int nvertices_per_instance, bool append);

class H5PartDump {
  float origin[3];
  void * handler;
  bool disposed;
  int tstamp;
  void _initialize(const std::string filename);
  void _dispose();
 public:
  H5PartDump(const std::string filename);
  void close() { _dispose(); }
  void dump(Particle * host_particles, int n);
  ~H5PartDump();
};

class H5FieldDump {
  static bool directory_exists;
  int last_idtimestep, globalsize[3];
  void _write_fields(const char * const path2h5,
                     const float * const channeldata[], const char * const * const channelnames, const int nchannels);
  void _xdmf_header(FILE * xmf);
  void _xdmf_grid(FILE * xmf, const char * const h5path, const char * const * channelnames, int nchannels);
  void _xdmf_epilogue(FILE * xmf);
public:
  H5FieldDump();
  void dump(Particle * p, int n);
  void dump_scalarfield(float * data, const char *channelname);
};
