namespace x {
static void i2d(int i, /**/ int d[3]) { /* fragment id to directiron */
    enum {X, Y, Z};
    d[X] = (i     + 2) % 3 - 1;
    d[Y] = (i / 3 + 2) % 3 - 1;
    d[Z] = (i / 9 + 2) % 3 - 1;
}
}
