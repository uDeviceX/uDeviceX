namespace rex {
void fin() {
    MC(l::m::Comm_free(&cart));

    delete packstotalstart;
    delete host_packstotalstart;
    delete host_packstotalcount;

    delete packscount;
    delete packsstart;
    delete packsoffset;
    delete packbuf;
    delete host_packbuf;

    for (int i = 0; i < SE_HALO_SIZE; i++) delete local[i];
    for (int i = 0; i < SE_HALO_SIZE; i++) delete remote[i];
}
}
