# GPU MapReduce
## About
An implementation of MapReduce that runs on GPUs using CUDA. This implementation is currently for performing word count on a large document.
We take advantage of the massive parallelism offered by the CUDA programming model to execute map, sort and reduce tasks in parallel on thousands of GPU threads.

Many design aspects of our implementation are based on the [Mars project](https://github.com/wenbinf/Mars). Compared to the original Mars implementation, we have added an improved design for sorting and grouping of string data which improves performance and resource consumption significantly.

Running this requires an Nvidia GPU which supports CUDA version >=12.4.

## how to run
### Option 1: Run on machine with CUDA installed
#### Pre-requisites:
 - Make sure that Nvidia GPU drivers are installed for your GPU
 - Make sure that CUDA version >=12.4 is installed on your machine

#### Build
```

# build the original Mars implementation
cd ./src/Mars-WordCount
make clean && make


# build our GPU Mapreduce version
cd ./src/WordCount
make
```

#### Generating the data set
The data generation script will generate one text file of size 256MB, 512MB, 1GB, 1.5GB, 2GB and 3GB each. Make sure that you have enough disk space to store all these files. Comment out the lines in the generateData.sh script to not generate files which you don't want.
```
cd ./data
chmod +x ./generateData.sh
./generateData.sh
```


#### Run
```
# run the word count
cd ./src/WordCount
./wordcount /path/to/data/file.txt



# run the Mars version for comparison
cd ./src/Mars-WordCount
./WordCount/WordCount /path/to/data/file.txt

```
***
### Option 2: Run in Docker container
#### Pre-requisites
 - On Linux, install Docker
 - On Windows, setup WSL Ubuntu
    ```
    # https://learn.microsoft.com/en-us/windows/wsl/setup/environment
    wsl -d Ubuntu
    ```
    Then install Docker in the WSL instance.
 - Install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
 
#### Build
```
# build the container image
sudo docker build . -t cuda

# create directory which will serve as a volume for the container
mkdir app

# copy the code into the volume
cp -r ./src ./app/
cp -r ./data ./app/

# run the container
sudo docker run --gpus all -it --rm -v $(pwd)/app:/app cuda bash
```
In the shell inside the container:
```
# build the Mars version
cd ./src/Mars-WordCount
make clean && make


# build our GPU Mapreduce version
cd ./src/WordCount
make
```
#### Generating the data set
The data generation script will generate one text file of size 256MB, 512MB, 1GB, 1.5GB, 2GB and 3GB each. Make sure that you have enough disk space to store all these files. Comment out the lines in the generateData.sh script to not generate files which you don't want.
```
# In the shell inside the container:

cd ./data
chmod +x ./generateData.sh
./generateData.sh
```

#### Run
```
# In the shell inside the container:

# run the word count
cd ./src/WordCount
./wordcount /path/to/data/file.txt



# run the Mars version for comparison
cd ./src/Mars-WordCount
./WordCount/WordCount /path/to/data/file.txt

```
## examples
```
❯ ./wordcount ../../data/data-1gb.txt 
            convert to upper case [CPU]:                409.519000ms
                split file blocks [CPU]:                14.288000ms
        copy input to device [CPU->GPU]:                326.816000ms
        calculate map output size [GPU]:                824.687000ms
     calculate intermediate sizes [GPU]:                61.208000ms
                              map [GPU]:                104.900000ms
     copy key-values to host [GPU->CPU]:                241.352000ms
                  sort key-values [GPU]:                760.864000ms
                    reduce by key [GPU]:                64.226000ms
copy final wordcount to host [GPU->CPU]:                0.024000ms

Number of unique words: 5543

First 10 words (alphabetical order):

ABANDON | Wordsize: 8 | Wordcount: 4868
ABANDONED | Wordsize: 10 | Wordcount: 7300
ABHORRED | Wordsize: 9 | Wordcount: 29196
ABHORRENCE | Wordsize: 11 | Wordcount: 14600
ABHORRENT | Wordsize: 10 | Wordcount: 2432
ABILITY | Wordsize: 8 | Wordcount: 4872
ABJECT | Wordsize: 7 | Wordcount: 4864
ABOARD | Wordsize: 7 | Wordcount: 2432
ABORTION | Wordsize: 9 | Wordcount: 2432
ABORTIVE | Wordsize: 9 | Wordcount: 2436

                           overall time:                3353.208000ms

```
