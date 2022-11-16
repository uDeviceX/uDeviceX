namespace bop_utils {

BopStatus safe_malloc(size_t sz, void **data);
BopStatus safe_open(const char *fname, const char *mode, FILE **f);

size_t get_bsize(BopType t);

void get_path(const char *full, char *path);
void get_fname_values(const char *fnbop, char *fnval);

const char * type2str(BopType t);
BopType str2type(const char *str);

void report_err(const char *fmt, ...);

} // bop_utils
