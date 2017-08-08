inline int zero0(dim3 i, dim3 j) {
    MSG("ij: %d %d", i.x*i.y*i.z, j.x*j.y*j.z);
    return 1;
}

inline int zero(dim3 i, dim3 j)        { return zero0(i, j); }
inline int zero(dim3 i, dim3 j, int n) { return zero0(i, j); }

#define KL_BEFORE(s, C) if (zero(C)) continue;
#define KL_AFTER(s) MSG("safe: %s", s);
#define KL_CALL(F, C, A) F<<<ESC C>>>A
