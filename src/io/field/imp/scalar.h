void scalar(Coords coords, MPI_Comm cart, float *D, const char *name) {
    char path[BUFSIZ];
    sprintf(path, DUMP_BASE "/h5/%s.h5", name);
    UC(h5::write(coords, cart, path, &D, &name, 1));
    if (m::is_master(cart)) xmf_write(path, &name, 1, xs(coords), ys(coords), zs(coords));
}
