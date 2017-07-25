namespace rex {
void fin() {
    delete packbuf;
    delete host_packbuf;

    for (int i = 0; i < 26; i++) delete local[i];
    for (int i = 0; i < 26; i++) delete remote[i];
}
}
