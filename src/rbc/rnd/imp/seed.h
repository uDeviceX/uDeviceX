static long decode_seed_env() {
    char *s;
    s = getenv("RBC_RND");
    if   (s == NULL) {
        msg_print("RBC_RND is not set");
        return 0;
    }
    else  {
        msg_print("RBC_RND = %s", s);
        return atol(s);
    }
}
static long decode_seed_time() {
    long t;
    t = os::time();
    msg_print("t: %ld", t);
    return t;
}
static int decode_seed(long seed) {
    if      (seed == ENV )  return decode_seed_env();
    else if (seed == TIME) return decode_seed_time();
    else return seed;
}
