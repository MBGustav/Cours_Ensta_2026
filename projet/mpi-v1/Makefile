include Make_linux.inc

# Configurações de Compilação
# -g para símbolos, -O3 para otimização, -fno-omit-frame-pointer para FlameGraph
CXXFLAGS = -std=c++17 -O3 -march=native -Wall $(MPIFLAGS)
EXEC = ant_simu.exe
NP = 4  # Número de processos MPI padrão

export PATH := $(PATH):/home/mgustav/git-repos/FlameGraph

ALL = $(EXEC)

default: help

all: $(ALL)

clean:
	@rm -fr *.o *.exe *~ *.out *.data *.data.* timechart.svg output.png flamegraph.svg

# Regra de compilação corrigida
.cpp.o:
	$(MPICXX) $(CXXFLAGS) -c $< -o $@  

# Linkagem MPI
$(EXEC): ant.o fractal_land.o renderer.o window.o ant_simu.o
	$(MPICXX) $(CXXFLAGS) $^ -o $@ $(LIB) -lpthread -lSDL2

# EXECUÇÃO E PROFILING MPI
# Nota: Usamos 'sudo perf record -a' para capturar todos os processos do nó
perf.data: $(EXEC)
	sudo rm -f perf.data
	@echo "Executando MPI com perf (monitorando todos os processos)..."
	perf record -g -F 999 -- mpirun -np $(NP) ./$(EXEC)

profile: perf.data
	@echo "Gerando Dot Graph..."
	sudo perf script | \
         c++filt | \
         grep -vE 'libgomp|pthread|libc|linux|\[.*\]|openmpi|mca' | \
         gprof2dot -f perf --show-samples | \
         dot -Tpng -o output.png


run: $(EXEC)
	@echo "Executando MPI com $(NP) processos..."
	mpiexec -n $(NP) ./$(EXEC)

profile_fg: perf.data
	@echo "Gerando Flamegraph MPI..."
	sudo perf script -f | \
    grep -vE 'libgomp|mca|\[.*\]' | \
    stackcollapse-perf.pl | \
    flamegraph.pl --inverted --minwidth=2 > flamegraph.svg

# TIMECHART PARA MPI
# Mostra a atividade de todos os cores durante a execução distribuída
timechart: $(EXEC)
	@echo "Limpando e configurando permissões..."
	sudo rm -f perf.data timechart.svg
	sudo sysctl -w kernel.perf_event_paranoid=-1
	sudo sysctl -w kernel.kptr_restrict=0
	@echo "Rodando MPI + Timechart..."
	# Gravamos o nó inteiro (-a) para ver como o MPI distribui a carga nos cores
	sudo perf record -a -e sched:sched_switch -e sched:sched_wakeup -e power:cpu_idle -- mpirun -np $(NP) --oversubscribe ./$(EXEC)
	@echo "Gerando SVG..."
	sudo perf timechart --power -o timechart.svg
	sudo chown $(USER):$(USER) timechart.svg
	chmod 644 timechart.svg

help:
	@echo "Targets disponíveis:"
	@echo "  all        : Compila o projeto com MPI"
	@echo "  timechart  : Analisa ocupação dos cores e IDLE em execução MPI"
	@echo "  profile    : Gera grafo de chamadas (output.png)"
	@echo "  profile_fg : Gera FlameGraph (flamegraph.svg)"
	@echo "Configuração atual: NP=$(NP) processos"