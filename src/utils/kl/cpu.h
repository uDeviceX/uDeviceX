#define KL_BEFORE(s, C) msg_print("kl: %s", s); if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F A
#define KL_AFTER(s)

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
