inline int zero(int i, int j) {
    return 1;
}

#define KL_BEFORE(i, j, ...) if (zero(i, j)) continue;
#define KL_AFTER(s) MSG("safe: %s", s);
#define KL_CALL(F, C, A) F<<<ESC C>>>A
