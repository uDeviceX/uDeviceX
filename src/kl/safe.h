inline int good0(const char s[], dim3 i, dim3 j) {
    int rc;
    rc = i.x>0 && i.y>0 && i.z>0 &&
         j.x>0 && j.y>0 && j.z>0;
    if (!rc) ERR("s: %s", s);
    return rc;
}
inline int good(const char s[], dim3 i, dim3 j)                    { return good0(s, i, j); }
inline int good(const char s[], dim3 i, dim3 j, int)               { return good0(s, i, j); }
inline int good(const char s[], dim3 i, dim3 j, int, cudaStream_t) { return good0(s, i, j); }

#define KL_BEFORE(s, C) if (!good(s, ESC C)) continue;
#define KL_AFTER(s) CC(cudaPeekAtLastError())
#define KL_CALL(F, C, A) F<<<ESC C>>>A
