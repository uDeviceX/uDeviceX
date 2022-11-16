static long decode_seed_time() {
    long t;
    t = os_time();
    msg_print("seed time: %ld", t);
    return t;
}
static int decode_seed(long seed) {
    if (seed == SEED_TIME) return decode_seed_time();
    else return seed;
}
