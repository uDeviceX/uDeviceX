class H5FieldDump {
    static bool directory_exists;
    void fields(const char * const path2h5,
                       const float * const channeldata[], const char * const * const channelnames, const int nchannels);
    void grid(FILE * xmf, const char * const h5path, const char * const * channelnames, int nchannels);
public:
    void dump(Particle * p, int n);
    void scalar(float * data, const char *channelname);
};
