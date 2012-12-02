all:
	$(MAKE) all -C dgemm
	$(MAKE) all -C sgemm

clean:
	$(MAKE) clean -C dgemm
	$(MAKE) clean -C sgemm

test:
	$(MAKE) test -C dgemm
	$(MAKE) test -C sgemm
