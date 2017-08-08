inline int zero0(dim3 i, dim3 j) {
<<<<<<< HEAD
    int n, m;
    n = i.x*i.y*i.z;
    m = j.x*j.y*j.z;
    MSG("ij: %d %d", n, m);
    return n > 0 && m > 0;
=======
    fprintf(stderr, "ij: %d %d\n", i.x*i.y*i.z, j.x*j.y*j.z);
    return 1;
>>>>>>> 15545b18e9e1714c5b13d14d9b1929078f2ebdc6
}
inline int zero(dim3 i, dim3 j)      { return zero0(i, j); }
inline int zero(dim3 i, dim3 j, int) { return zero0(i, j); }

#define KL_BEFORE(s, C) if (zero(C)) continue;
#define KL_AFTER(s) MSG("safe: %s", s);
#define KL_CALL(F, C, A) F<<<ESC C>>>A
