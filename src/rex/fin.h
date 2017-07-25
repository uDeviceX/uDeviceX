namespace rex {
void fin() {
    delete packstotalstart;
    delete host_packstotalstart;
    delete host_packstotalcount;

    delete packscount;
    delete packsstart;
    delete packsoffset;
    delete packbuf;
    delete host_packbuf;

    for (int i = 0; i < 26; i++) delete local[i];
    for (int i = 0; i < 26; i++) delete remote[i];
}
}
