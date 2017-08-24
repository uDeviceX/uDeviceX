namespace dbg {
#define DBG_BEFORE(s, M) err::ini()
#define DBG_CALL(F, A) F A;
#define DBG_AFTER(s) err::handle()
} // dbg
