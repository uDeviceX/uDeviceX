# forces::Fo

`Fo` is an abstract force returned by `forces::gen()`.
clean, Supported operations are

	clean(Fo *a)
	plus (Fo *a, /**/ Fo *b)

analogies to

	a  = 0
	b += a

# toocha

Toocha (heavy cloud in Russian) is an abstract object. It act like a
pointer to `Fo`. Supported operations are

	shift(int i, /**/ Toocha *t)

	plus  (Fo f, /**/ Toocha t)
	minus (Fo f, /**/ Toocha t)

analogies to

	t += i

	(*t) += f
	(*t) -= f
