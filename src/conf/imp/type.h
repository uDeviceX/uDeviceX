// tag::struct[]
enum {
    EXE, /* from program setters */
    ARG, /* from arguments       */
    OPT, /* from additional file */
    DEF, /* from default file    */
    NCFG
};

struct Config {
    config_t c[NCFG];
    config_t *r; /* what have been read */
};
// end::struct[]

enum {
    OK,
    NOTFOUND,
    WTYPE
};

enum {
    MAX_LEVEL = 10 /* maximum level of parameters */
};

struct CBuf {
    char c[MAX_LEVEL][FILENAME_MAX];
};
