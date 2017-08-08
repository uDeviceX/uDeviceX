#define ESC(...) __VA_ARGS__
#define KL_BEFORE(...)
#define KL_AFTER(s)
#define KL_CALL(F, C, A) F<<<ESC C>>>A

#define KL(F, C, A)                            \
    do  {                                       \
        KL_BEFORE(#F, ESC C);                   \
        KL_CALL(F, C, A);                       \
        KL_AFTER(#F);                           \
    } while (0)

/* examples */
/* ceiling `m' to `n' (returns the smallest `A' such n*A is not less than `m') */
#define ceiln(m, n) (   ((m) + (n) - 1)/(n)   )
/* a common kernel execution configuration */
#define k_cnf(n) ceiln((n), 128), 128

KL(fun, (i+1, j, k), ())
KL(fun, (i+1, j, k), (a, b))
KL(dev::update, (k_cnf(o::q.n)), (dpd_mass, o::q.pp, o::ff, o::q.n));

/* run the command to expand

$ cpp kl.h  | grep -v '^#'

*/
