#define KL_BEFORE(s, C) MSG("kl: %s", s); if (!kl::safe(ESC C)) continue;
#define KL_AFTER(s) CC(cudaPeekAtLastError());
#define KL_CALL(F, C, A) F<<<ESC C>>>A

namespace kl {
inline void msg(int ix, int iy, int iz,   int jx, int jy, int jz) {
    MSG("klconf: [%d %d %d] [%d %d %d]", ix, iy, iz,   jx, jy, jz);
};
}
