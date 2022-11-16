enum {MAX_NK = 100 };
enum {STAMP_GOOD = 42 };

struct KeyList {
    int nk; /* number of keys */
    int  ww[MAX_NK]; /* width */
    int  mark[MAX_NK];
    char keys[MAX_NK][FILENAME_MAX];
    int stamp;
};
