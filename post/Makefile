
subdirs=$(shell find . -mindepth 1 -maxdepth 1 -type d)
deps=$(subdirs:%=%.dummy)

install: $(deps)

%.dummy: %
	@make install -C $<
