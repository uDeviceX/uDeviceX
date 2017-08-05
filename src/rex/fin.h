namespace rex {
void fin() {
    for (int i = 0; i < 26; i++) delete local[i];
    for (int i = 0; i < 26; i++) delete remote[i];
}
}
