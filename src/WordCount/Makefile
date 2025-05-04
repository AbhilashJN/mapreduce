CUFLAGS     := -O2 --compiler-options '-fPIC'

all:
	make clean && make wordcount

clean:
	rm -rf wordcount

wordcount:
	nvcc $(CUFLAGS) -o wordcount main.cu mapreduce.cu host_utils.cu device_utils.cu