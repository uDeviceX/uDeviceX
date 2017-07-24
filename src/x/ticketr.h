namespace x {
static void ini_ticketr(TicketR *t) {
    enum {X, Y, Z};
    int i;
    int d[3];
    for (i = 0; i < 26; ++i) {
        i2d(i, /**/ d);
        t->tags[i] = \
                 (2 - d[X]) % 3 +
            3 * ((2 - d[Y]) % 3 +
            3 * ((2 - d[Z]) % 3));
    }
}
}
