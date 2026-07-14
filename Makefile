# Non-temporal store stencil benchmark.
#   make            build ./stencil
#   make verify-asm confirm the *_nt kernels really emit STNP (Apple Silicon)
#   make clean
#
# Apple Silicon needs Homebrew's libomp:  brew install libomp

CXXFLAGS := -std=c++17 -O3 -ffast-math -Wall -Wextra

ifeq ($(shell uname -s),Darwin)
  CXX      := clang++
  CXXFLAGS += -mcpu=apple-m2 -DUSE_GCD
  OMP      := -Xpreprocessor -fopenmp -I$(shell brew --prefix libomp)/include
  OMPLIB   := -L$(shell brew --prefix libomp)/lib -lomp
else
  CXX      := g++
  CXXFLAGS += -march=native
  OMP      := -fopenmp
  OMPLIB   := -fopenmp
endif

stencil: stencil.cpp
	$(CXX) $(CXXFLAGS) $(OMP) stencil.cpp -o $@ $(OMPLIB)

verify-asm: stencil.cpp
	@$(CXX) $(CXXFLAGS) $(OMP) -S stencil.cpp -o stencil.s
	@echo "STNP (non-temporal pair store) count per kernel:"
	@for fn in k_jacobi_std k_jacobi_nt k_nine_std k_nine_nt; do \
	  printf "  %-14s %s\n" "$$fn" \
	    "$$(awk "/^_?$$fn:/{f=1} f{print} f&&/\.cfi_endproc/{exit}" stencil.s | grep -ci stnp)"; \
	done

clean:
	rm -f stencil stencil.s

.PHONY: verify-asm clean
