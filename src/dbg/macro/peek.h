namespace dbg {
#define DBG_BEFORE(s, file, line, M) err::ini()
#define DBG_CALL(F, A) F A;
#define DBG_AFTER(s, file, line, M) err::handle(line, file, s, M)
} // dbg
