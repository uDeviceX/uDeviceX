void H5FieldDump::scalar(float *D, const char *name) {
    char path[BUFSIZ];
    sprintf(path, DUMP_BASE "/h5/%s.h5", name);
    fields(path, &D, &name, 1);
}
