# Paths related to LibOPT
OPT_HOME=/home/lzfelix/work/LibOPT
LIB=$(OPT_HOME)/lib
INCLUDE=$(OPT_HOME)/include
OBJ=$(OPT_HOME)/obj

# Where the target files are
INCLUDE_CUSTOM=./lib
SRC=./src
BIN=./bin

# Compilation flags. You might want to change the compiler below
CC=gcc-7
DEBUG_FLAGS=-g
FLAGS=-O3 -Wno-unused-result

# Path to Valgrind, used only for debugging
VALGRIND=valgrind

# Path to the Python installation to be used for F1 maximization. Ensure
# to have the python3.6-dev headers installed. This can be done with the
# following lines:
#   sudo add-apt-repository ppa:deadsnakes/ppa
#   sudo apt-get update
#   sudo apt-get install python3.6-dev
PYTHON_INCLUDE=/home/lzfelix/bin/miniconda2/envs/dev/include/python3.6m
PYTHON_PATH=/home/lzfelix/bin/miniconda2/envs/dev/bin/python3.6-config

# If the compilation process works fine, then you don't need to change
# the lines below.
PYTHON_LINK_FLAGS=$(shell $(PYTHON_PATH) --ldflags)
PYTHON_COMPILE_FLAGS=$(shell $(PYTHON_PATH) --cflags)
# PYTHON_COMPILE_FLAGS="-I/home/lzfelix/bin/miniconda2/envs/dev/include/python3.6m -I/home/lzfelix/bin/miniconda2/envs/dev/include/python3.6m  -Wno-unused-result -Wsign-compare -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -O3 -pipe  -fdebug-prefix-map==/usr/local/src/conda/- -fdebug-prefix-map==/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall -Wstrict-prototypes"


all: build

build:
	echo $(PYTHON_COMPILE_FLAGS)\n;

    # Compiling GA
	$(CC) $(FLAGS) $(SRC)/ga.c -o $(BIN)/optimizer_ga -std=c99 \
    -I $(INCLUDE) -I $(INCLUDE_CUSTOM) -I $(PYTHON_INCLUDE) -I $(PYTHON_COMPILE_FLAGS) \
    -L $(LIB) -L $(PYTHON_LINK_FLAGS) -lopt -lm;
    
	$(CC) $(FLAGS) $(SRC)/gp.c -o $(BIN)/optimizer_gp -std=c99 \
    -I $(INCLUDE) -I $(INCLUDE_CUSTOM) -I $(PYTHON_INCLUDE) -I $(PYTHON_COMPILE_FLAGS) \
    -L $(LIB) -L $(PYTHON_LINK_FLAGS) -lopt -lm;

valgrind_ga:
	$(CC) $(DEBUG_FLAGS) $(SRC)/ga.c -o bin/optimizer_ga -std=c99 \
    -I $(INCLUDE) -I $(INCLUDE_CUSTOM) -I $(PYTHON_INCLUDE) -I $(PYTHON_COMPILE_FLAGS) \
    -L $(LIB) -L $(PYTHON_LINK_FLAGS) -lopt -lm;

	$(VALGRIND) --leak-check=yes ./bin/optimizer_ga softmax_ds3 > /dev/null

valgrind_gp:
	$(CC) $(DEBUG_FLAGS) $(SRC)/gp.c -o bin/optimizer_gp -std=c99 \
    -I $(INCLUDE) -I $(INCLUDE_CUSTOM) -I $(PYTHON_INCLUDE) -I $(PYTHON_COMPILE_FLAGS) \
    -L $(LIB) -L $(PYTHON_LINK_FLAGS) -lopt -lm;

	$(VALGRIND) --leak-check=yes ./bin/optimizer_gp softmax_ds3 > /dev/null
