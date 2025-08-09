all:
	nvcc main.cu kmer_serial.cu kmer_serial_bm.cu kmer_parallel.cu -o main
	gcc -fopenmp kmer_cpu_parallel.c -o kmer_cpu_parallel

clean:
	rm -f main kmer_cpu_parallel
