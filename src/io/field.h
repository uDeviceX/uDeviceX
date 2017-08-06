class H5FieldDump {
    static bool directory_exists;
    int globalsize[3];
    void fields(const char * const path2h5,
                       const float * const channeldata[], const char * const * const channelnames, const int nchannels);
    void header(FILE * xmf);
    void grid(FILE * xmf, const char * const h5path, const char * const * channelnames, int nchannels);
    void epilogue(FILE * xmf);
public:
    H5FieldDump();
    void dump(Particle * p, int n);
    void scalar(float * data, const char *channelname);
};
