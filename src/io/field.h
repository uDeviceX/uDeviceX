class H5FieldDump {
    static bool directory_exists;
public:
    void dump(Particle * p, int n);
    void scalar(float * data, const char *channelname);
};
