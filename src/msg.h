#define MSG00(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#define MSG(fmt, ...) MSG00("%03d: ", m::rank), MSG00(fmt, ##__VA_ARGS__), MSG00("\n")

#define MSG0(fmt, ...)                             \
    do {                                           \
        if (m::rank == 0) MSG(fmt, ##__VA_ARGS__); \
    } while (0)

#define ERR(fmt, ...)                                                   \
    do {                                                                \
        fprintf(stderr, "%03d: ERROR: %s:%d: " fmt,                     \
                m::rank, __FILE__, __LINE__, ##__VA_ARGS__);            \
        exit(1);                                                        \
    } while(0)
