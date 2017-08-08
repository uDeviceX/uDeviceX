#define KL_BEFORE(...)
#define KL_AFTER(s) MSG("safe: %s", s);
#define KL_CALL(F, C, A) F<<<ESC C>>>A
