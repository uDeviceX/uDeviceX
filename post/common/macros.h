#define ERR(...) do {                           \
        fprintf(stderr, __VA_ARGS__);           \
        exit(1);                                \
    } while (0);

/* BOV check */

#define BVC(ans) do {                           \
        BovStatus s = (ans);                    \
        if (!bov_success(s)) {                  \
            ERR(":%s:%d: %s\n%s\n",             \
                __FILE__, __LINE__,             \
                bov_report_error_desc(s),       \
                bov_report_error_mesg());       \
            exit(1);                            \
        }} while (0)

/* BOP check */

#define BPC(ans) do {                           \
        BopStatus s = (ans);                    \
        if (!bop_success(s)) {                  \
            ERR(":%s:%d: %s\n%s\n",             \
                __FILE__, __LINE__,             \
                bop_report_error_desc(s),       \
                bop_report_error_mesg());       \
        }} while (0)
