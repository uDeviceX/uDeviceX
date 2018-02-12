void scalar(const Coords *coords, MPI_Comm cart, float *D, const char *name) {
    char path[BUFSIZ];
    sprintf(path, DUMP_BASE "/h5/%s.h5", name);
    if (m::is_master(cart)) UC(os_mkdir(DUMP_BASE "/h5"));
    UC(h5_write(coords, cart, path, &D, &name, 1));
    if (m::is_master(cart)) xmf_write(coords, path, &name, 1);
}
