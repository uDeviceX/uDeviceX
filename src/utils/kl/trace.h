#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s) msg_print("kl: %s", s);

namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
