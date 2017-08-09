#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) ERR("unsafe kernel call: %s", s);
#define KL_AFTER(s)
#define KL_CALL(F, C, A) F<<<ESC C>>>A
