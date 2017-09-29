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

	plus  (Fo f, int i, /**/ Toocha t)
	minus (Fo f, int i, /**/ Toocha t)

analogies to

	t += i

	(*t) += f
	(*t) -= f

	t[i] += f
	t[i] -= f

Toocha can be build from `Force*` and `Stress*`

	ini_toocha(Force *ff,             /**/ Toocha *t)
	ini_toocha(Force *ff, Stress *ss, /**/ Toocha *t)
