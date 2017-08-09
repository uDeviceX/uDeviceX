#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_AFTER(s) CC(cudaPeekAtLastError());
#define KL_CALL(F, C, A) F<<<ESC C>>>A
