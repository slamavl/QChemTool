.PHONY: all clean test

all:
	env F90FLAGS="-fopenmp -ffree-line-length-none" python3 setup.py build_ext --inplace --fcompiler=gfortran 

test: all
	@echo Running with 1 thread...
	env OMP_NUM_THREADS=1 python test.py
	@echo Running with 2 thread...
	env OMP_NUM_THREADS=2 python test.py
	@echo Running with 4 thread...
	env OMP_NUM_THREADS=4 python test.py
	@echo Running with 8 thread...
	env OMP_NUM_THREADS=8 python test.py

clean:
	rm -rf build
	rm -rf wrapper.so
