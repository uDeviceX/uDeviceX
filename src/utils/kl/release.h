#define KL_BEFORE(s, C) if (!kl::safe(ESC C)) continue;
#define KL_CALL(F, C, A) F<<<ESC C>>>A
#define KL_AFTER(s)

#ifndef UDX_KL_RELEASE_FUNCTIONS_DEFINED
#define UDX_KL_RELEASE_FUNCTIONS_DEFINED
namespace kl { inline void msg(int, int, int,   int, int, int) { }; }
#endif
