CXX := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic
# OpenMP parallelizes the rotating solver's residual evaluation (make OPENMP=0 to disable)
OPENMP ?= 1
ifeq ($(OPENMP),1)
CXXFLAGS += -fopenmp
endif
# Vendored header-only Eigen (third_party/eigen/VENDORED.txt); -isystem silences its warnings
INCLUDES := -isystem third_party/eigen
# -MMD -MP: emit .d files so objects rebuild when included headers change
DEPFLAGS := -MMD -MP

SRCS := main.cpp params_io.cpp solver.cpp mathutils.cpp linear_solvers.cpp potentials.cpp rotating_solver.cpp
OBJS := $(SRCS:.cpp=.o)
DEPS := $(SRCS:.cpp=.d)
TARGET := rotating_bs_solver

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEPFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) rotating_params.par

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

-include $(DEPS)
