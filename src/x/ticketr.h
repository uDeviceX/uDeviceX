namespace x {
/* ticket recive */
static void ini_ticketr(TicketR *t) {
    enum {X, Y, Z};
    int i;
    for (i = 0; i < 26; ++i) {
        const int *d = frag_to_dir[i];
        t->tags[i] = \
                 (2 - d[X]) % 3 +
            3 * ((2 - d[Y]) % 3 +
            3 * ((2 - d[Z]) % 3));
    }
}
}
