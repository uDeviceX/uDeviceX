# forces::Fo

`Fo` is an abstract force returned by `forces::gen()`.  Supported
operations are

	clean(Fo *a)
	plus (Fo *a, /**/ Fo *b)

analogies to

	a  = 0
	b += a

# toocha

Toocha (heavy cloud in Russian) is an abstract object which acts like
a pointer to `Fo`. Supported operations are

	shift(int i, /**/ Toocha *t)

	plus0 (Fo f, /**/ Toocha t)
	minus0(Fo f, /**/ Toocha t)

	plus  (Fo f, /**/ Toocha t, int i)
	minus (Fo f, /**/ Toocha t, int i)

analogies to

	t += i

	(*t) += f
	(*t) -= f

	t[i] += f
	t[i] -= f
