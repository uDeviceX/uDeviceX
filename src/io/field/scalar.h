void H5FieldDump::scalar(float *data,
                                   const char *channelname) {
    char path2h5[512];
    sprintf(path2h5, DUMP_BASE "/h5/%s.h5", channelname);
    fields(path2h5, &data, &channelname, 1);
}
