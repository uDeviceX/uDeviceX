#define KL_BEFORE(s, C) msg_print("kl: %s", s); if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s) CC(d::PeekAtLastError());

namespace kl {
inline void msg(int ix, int iy, int iz,   int jx, int jy, int jz) {
    msg_print("klconf: [%d %d %d] [%d %d %d]", ix, iy, iz,   jx, jy, jz);
};
}
