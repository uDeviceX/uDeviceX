namespace msg {
extern char buf[];
void print();
}

#define MSG(fmt, ...)                             \
    do {                                          \
        sprintf(msg::buf, fmt, ##__VA_ARGS__);    \
        msg::print();                             \
    } while (0)

#define ERR(fmt, ...)                                                   \
    do {                                                                \
        MSG(fmt, ##__VA_ARGS__);                                        \
        exit(1);                                                        \
    } while(0)
