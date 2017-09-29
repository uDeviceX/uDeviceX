# forces::Fo

`Fo` is an abstract force returned by `forces::gen()`.
clean, plus, and minus are the supported operations:

	clean(Fo *a)
	plus (Fo *a, /**/ Fo *b)
	minus(Fo *a, /**/ Fo *b)

analogies to

	a = 0
	b += a
	b -= a

# toocha

Toocha (heavy cloud in Russian) is an abstract object. It act like a
pointer to an array of `Fo`. inc, set are the supported operations

	inc(Toocha *t, int i)
	set(Toocha  t, /**/ Fo f)
	
analogies to
   
    t += i
	*t = f
