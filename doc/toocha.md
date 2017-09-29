# forces::Fo

`Fo` is an abstract force returned by `forces::gen()`.
clean, Supported operations are

	clean(Fo *a)
	plus (Fo *a, /**/ Fo *b)

analogies to

	a = 0
	b += a

# toocha

Toocha (heavy cloud in Russian) is an abstract object. It act like a
pointer to `Fo`. Supported operations are

	shift(Toocha *t, int i)

	plus (Toocha  t,  Fo f)
	minus(Toocha  t,  Fo f)

analogies to

	t += i
	
	(*t) += f
	(*t) -= f	
