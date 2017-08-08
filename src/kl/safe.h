inline int zero(dim3 i, dim3 j) {
    return 1;
}

#define KL_BEFORE(name, i, j, ...) if (zero(i, j)) continue;
#define KL_AFTER(s) MSG("safe: %s", s);
#define KL_CALL(F, C, A) F<<<ESC C>>>A
