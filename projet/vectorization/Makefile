include Make_linux.inc
#include Make_msys2.inc
#include Make_osx.inc

CXXFLAGS = -std=c++17 -fno-omit-frame-pointer
EXEC=ant_simu.exe

export PATH := $(PATH):/home/mgustav/git-repos/FlameGraph

ifdef DEBUG
CXXFLAGS += -pg -O0 -Wall -fbounds-check -pedantic -D_GLIBCXX_DEBUG
CXXFLAGS2 = $(CXXFLAGS)
else
CXXFLAGS2 = ${CXXFLAGS} -O2 -march=native -Wall 
CXXFLAGS += -g -O3 -march=native -Wall
endif

ALL= ant_simu.exe 

default:	help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~ *.out *.data *.data.* 
.cpp.o:
	$(CXX) $(CXXFLAGS2) -c $^ -o $@	

ant_simu.exe : ant.o fractal_land.o renderer.o window.o ant_simu.o
	$(CXX) $(CXXFLAGS2) $^ -o $@ $(LIB)	

help:
	@echo "Available targets : "
	@echo "    all            : compile all executables"
	@echo "Add DEBUG=yes to compile in debug"
	@echo "Configuration :"
	@echo "    CXX      :    $(CXX)"
	@echo "    CXXFLAGS :    $(CXXFLAGS)"

perf.data: $(EXEC)
	sudo rm -f perf.data timechart.svg
	@echo "Running program with perf..."
	sudo perf record -g -F 999 -- ./$(EXEC)

profile: perf.data
	@echo "Running program with perf..."
	sudo perf script 							 	| \
		 c++filt 							     	| \
		 grep -vE 'libgomp|pthread|libc|linux|\[.*\]' | \
		 gprof2dot -f perf --show-samples 	| \
		 dot -Tpng -o output.png

profile_fg: perf.data
	@echo "Processing Flamegraph..."
	sudo perf script                 | \
	grep -vE 'libgomp|libc|\[.*\]'   | \
	stackcollapse-perf.pl            | \
	flamegraph.pl --inverted --minwidth=2 > flamegraph.svg


timechart: $(EXEC)
	@echo "Cleaning previous data..."
	sudo rm -f perf.data timechart.svg
	
	@echo "System Permissions..."
	sudo sysctl -w kernel.perf_event_paranoid=-1
	sudo sysctl -w kernel.kptr_restrict=0
	
	sudo perf record -e sched:sched_switch -e sched:sched_wakeup -- ./$(EXEC)
	
	@echo "Generating timechart..."
	# IMPORTANTE: Geramos o timechart e alteramos o dono do arquivo 
	# para que você possa abri-lo sem ser root (evita erro de leitura).
	sudo perf timechart -o timechart.svg
	sudo chmod 644 timechart.svg
	sudo chown $(USER):$(USER) timechart.svg
	@echo "-----------------------------------------------------------"

%.html: %.md
	pandoc -s --toc $< --css=./github-pandoc.css --metadata pagetitle="OS202 - TD1" -o $@
