# Makefile for the non-temporal store stencil benchmark.
#
#   make            build ./stencil
#   make verify-asm show how many STNP (non-temporal) stores each kernel got
#   make sweep      build + run the size sweep in launch.sh
#   make clean
#
# On Apple Silicon this needs Homebrew's libomp:  brew install libomp
# Override any variable on the command line, e.g.  make CXX=clang++ ARCH=-mcpu=apple-m1

UNAME_S := $(shell uname -s)

BIN := stencil
SRC := stencil.cpp
CXXFLAGS ?= -std=c++17 -O3 -ffast-math -Wall -Wextra

ifeq ($(UNAME_S),Darwin)
  # --- Apple Silicon: clang++ + Homebrew libomp + Grand Central Dispatch ------
  # (override the built-in default of `c++`, but honour an explicit make CXX=...)
  ifeq ($(origin CXX),default)
    CXX := clang++
  endif
  ARCH     ?= -mcpu=apple-m2
  LIBOMP   := $(shell brew --prefix libomp 2>/dev/null)
  OMPFLAGS := -Xpreprocessor -fopenmp -I$(LIBOMP)/include
  OMPLIBS  := -L$(LIBOMP)/lib -lomp
  EXTRA    := -DUSE_GCD
else
  # --- Linux / other: g++ (or clang++) with its native OpenMP, no GCD ---------
  ifeq ($(origin CXX),default)
    CXX := g++
  endif
  ARCH     ?= -march=native
  OMPFLAGS := -fopenmp
  OMPLIBS  := -fopenmp
  EXTRA    :=
  # x86 clang can't vectorize the scalar NT store and warns; g++ doesn't know
  # the flag, so only pass it to a clang-based compiler.
  ifneq (,$(findstring clang,$(CXX)))
    EXTRA  += -Wno-pass-failed
  endif
endif

BUILD := $(CXX) $(CXXFLAGS) $(ARCH) $(OMPFLAGS) $(EXTRA)

$(BIN): $(SRC) Makefile
	$(BUILD) $(SRC) -o $@ $(OMPLIBS) -lm

# Emit assembly with the exact build flags and count STNP per kernel. The whole
# point of the benchmark rests on this: the compiler does NOT guarantee it will
# keep the non-temporal hint, so confirm the *_nt kernels show nonzero STNP
# before trusting any "nt" number. (x86 has no STNP; run this on Apple Silicon.)
.PHONY: verify-asm
verify-asm: $(SRC) Makefile
	@$(BUILD) -S $(SRC) -o $(BIN).s
	@echo "STNP (non-temporal pair store) count per kernel:"
	@for fn in k_jacobi_std k_jacobi_nt k_nine_std k_nine_nt; do \
	  n=$$(awk "/^_?$$fn:/{f=1} f{print} f&&/\.cfi_endproc/{exit}" $(BIN).s | grep -ci stnp); \
	  printf "  %-14s %s\n" "$$fn" "$$n"; \
	done
	@echo "(expect 0 for *_std, >0 for *_nt on Apple Silicon)"

.PHONY: sweep
sweep: $(BIN)
	./launch.sh

.PHONY: clean
clean:
	rm -f $(BIN) $(BIN).s *.o
