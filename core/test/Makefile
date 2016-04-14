# Author: Karl Stratos (stratos@cs.columbia.edu)

# Where to find the root of Google Test.
GTEST = ../../third_party/gtest-1.7.0

# Where to find the Eigen package.
EIGEN = ../../third_party/eigen-eigen-10219c95fe65

# Where to find the SVDLIBC package.
SVDLIBC = ../../third_party/SVDLIBC

# Where to find the core files.
CORE = ..

# C++ compiler.
CC = clang++

# Warning level.
WARN = -Wall

# Flags passed to the C++ compiler.
CFLAGS = -isystem $(GTEST)/include $(WARN) -std=c++11
ifeq ($(shell uname), Darwin)  # Apple clang version 4.0
	CFLAGS += -stdlib=libc++
endif

# Top-level commands.
TESTS = util_test eigen_helper_test optimize_test sparsesvd_test corpus_test \
	cluster_test evaluate_test trees_test features_test

all: $(TESTS)

.PHONY: clean
clean:
	rm -f $(TESTS) gtest.a gtest_main.a *.o $(CORE)/*.o
	make -C $(SVDLIBC) clean

############## Usually you shouldn't change this section. ######################
# All Google Test headers.
GTEST_HEADERS = $(GTEST)/include/gtest/*.h \
                $(GTEST)/include/gtest/internal/*.h

# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST)/src/*.cc $(GTEST)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
gtest-all.o: $(GTEST_SRCS_)
	     $(CC) -I $(GTEST) $(CFLAGS) -c $(GTEST)/src/gtest-all.cc

gtest_main.o: $(GTEST_SRCS_)
	      $(CC) -I $(GTEST) $(CFLAGS) -c $(GTEST)/src/gtest_main.cc

gtest.a: gtest-all.o
	 $(AR) $(ARFLAGS) $@ $^

gtest_main.a: gtest-all.o gtest_main.o
	      $(AR) $(ARFLAGS) $@ $^
################################################################################

# Builds tests.  A test should link with either gtest.a or gtest_main.a,
# depending on whether it defines its own main() function.

util_test: util_test.o $(CORE)/util.o gtest_main.a
	$(CC) $(CFLAGS) $^ -o $@ -pthread

util_test.o: util_test.cc
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/util.o: $(CORE)/util.cc $(CORE)/util.h
	$(CC) $(CFLAGS) -c $< -o $@

eigen_helper_test: eigen_helper_test.o $(CORE)/eigen_helper.o gtest_main.a
	$(CC) $(CFLAGS) $^ -o $@ -pthread

eigen_helper_test.o: eigen_helper_test.cc
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(CORE)/eigen_helper.o: $(CORE)/eigen_helper.cc $(CORE)/eigen_helper.h
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

optimize_test: optimize_test.o $(CORE)/optimize.o gtest_main.a
	$(CC) $(CFLAGS) $^ -o $@ -pthread

optimize_test.o: optimize_test.cc
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(CORE)/optimize.o: $(CORE)/optimize.cc $(CORE)/optimize.h
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

sparsesvd_test: sparsesvd_test.o $(CORE)/sparsesvd.o $(SVDLIBC)/libsvd.a \
	gtest_main.a
	$(CC) $(CFLAGS) $^ -o $@ -pthread

sparsesvd_test.o: sparsesvd_test.cc
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(CORE)/sparsesvd.o: $(CORE)/sparsesvd.cc $(CORE)/sparsesvd.h
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(SVDLIBC)/libsvd.a:
	make -C $(SVDLIBC)

corpus_test: corpus_test.o $(CORE)/corpus.o $(CORE)/util.o $(CORE)/sparsesvd.o \
	$(SVDLIBC)/libsvd.a $(CORE)/eigen_helper.o gtest_main.a
	$(CC) $(CFLAGS) $^ -o $@ -pthread

corpus_test.o: corpus_test.cc
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(CORE)/corpus.o: $(CORE)/corpus.cc $(CORE)/corpus.h $(CORE)/sparsesvd.o
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

cluster_test: cluster_test.o $(CORE)/cluster.o gtest_main.a
	$(CC) $(CFLAGS) $^ -o $@ -pthread

cluster_test.o: cluster_test.cc
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(CORE)/cluster.o: $(CORE)/cluster.cc $(CORE)/cluster.h
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

evaluate_test: evaluate_test.o $(CORE)/evaluate.o $(CORE)/util.o gtest_main.a
	$(CC) -I $(EIGEN) $(CFLAGS) $^ -o $@ -pthread

evaluate_test.o: evaluate_test.cc
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(CORE)/evaluate.o: $(CORE)/evaluate.cc $(CORE)/evaluate.h
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

trees_test: trees_test.cc $(CORE)/trees.o $(CORE)/util.o gtest_main.a
	$(CC) $(CFLAGS) $^ -o $@ -pthread

trees_test.o: trees_test.cc
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/trees.o: $(CORE)/trees.cc $(CORE)/trees.h $(CORE)/util.o
	$(CC) $(CFLAGS) -c $< -o $@

features_test: features_test.cc $(CORE)/features.o gtest_main.a
	$(CC) $(CFLAGS) $^ -o $@ -pthread

features_test.o: features_test.cc
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/features.o: $(CORE)/features.cc $(CORE)/features.h
	$(CC) $(CFLAGS) -c $< -o $@
