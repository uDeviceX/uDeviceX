#define KL_BEFORE(...)
#define KL_AFTER(s) MSG("kl: %s", s);
#define KL_CALL(F, C, A) F<<<ESC C>>>A
