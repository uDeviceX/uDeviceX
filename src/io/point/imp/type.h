struct BopData;
enum { N_MAX = 100 };

struct IOPointConf {
    int nn[N_MAX];
    char keys[N_MAX][FILENAME_MAX];
    int i;
};

struct IOPoint {
    int n;
    int nn[N_MAX];
    char keys[N_MAX][FILENAME_MAX];
    int maxn;
    char path[FILENAME_MAX];

    BopData *bop;
};
