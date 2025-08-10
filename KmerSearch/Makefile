all:
	nvcc -Xcompiler -fopenmp main.cu kmer_serial.cu kmer_serial_bm.cu kmer_parallel.cu kmer_cpu_parallel.cu -o main

run:
	./script.sh

clean:
	rm -f main out_*.txt