struct BopData;
enum { N_MAX = 100 };

struct IOPointConf {
    int nn[N_MAX];
    char keys[N_MAX][FILENAME_MAX];
    int i;
};

struct IOPoint {
    int maxn;
    int cum_n;

    int nkey;
    int nn[N_MAX];
    int seen[N_MAX];
    char keys[N_MAX][FILENAME_MAX];
    
    char path[FILENAME_MAX];
    BopData *bop;
};
