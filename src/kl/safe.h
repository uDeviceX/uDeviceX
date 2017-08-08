inline int zero(dim3 i, dim3 j) {
    return 1;
}

inline int zero(dim3 i, dim3 j, int n) {
    return 1;
}

#define KL_BEFORE(s, C) if (zero(C)) continue;
#define KL_AFTER(s) MSG("safe: %s", s);
#define KL_CALL(F, C, A) F<<<ESC C>>>A
