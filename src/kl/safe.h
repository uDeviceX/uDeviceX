inline int safe0(dim3 i, dim3 j) {
    return i.x>0 && i.y>0 && i.z>0   &&   j.x>0 && j.y>0 && j.z>0;
}
inline int safe(dim3 i, dim3 j)                    { return safe0(i, j); }
inline int safe(dim3 i, dim3 j, int)               { return safe0(i, j); }
inline int safe(dim3 i, dim3 j, int, cudaStream_t) { return safe0(i, j); }

#define KL_BEFORE(s, C) if (!safe(ESC C)) continue;
#define KL_AFTER(s)
#define KL_CALL(F, C, A) F<<<ESC C>>>A
