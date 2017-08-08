#define ESC(...) __VA_ARGS__
#define KL_BEFORE(...)
#define KL_AFTER(s)
#define KL_CALL(F, C, A) F<<<ESC C>>>A

#define KL(F, C, A)                             \
    do  {                                       \
        KL_BEFORE(#F, ESC C);                   \
        KL_CALL(F, C, A);                       \
        KL_AFTER(#F);                           \
    } while (0)

KL(fun, (i+1, j, k), ())
KL(fun, (i+1, j, k), (a, b))

/* run the command to expand

$ cpp kl.h  | grep -v '^#'

*/
