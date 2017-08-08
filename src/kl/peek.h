#define KL_BEFORE(...)
#define KL_AFTER(s) CC(cudaPeekAtLastError())
#define KL_CALL(F, C, A) F<<<ESC C>>>A
