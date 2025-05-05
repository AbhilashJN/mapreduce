#!/bin/bash

# generate a 256MB data file
for i in {1..1000};do cat frankenstein.txt >> data-256mb.txt; done 
truncate -s 256M data-256mb.txt

# generate a 512MB data file
cat data-256mb.txt >> data-512mb.txt
cat data-256mb.txt >> data-512mb.txt

# generate a 1GB data file
cat data-512mb.txt >> data-1gb.txt
cat data-512mb.txt >> data-1gb.txt


# generate a 1.5GB data file
cat data-1gb.txt >> data-1_5gb.txt
cat data-512mb.txt >> data-1_5gb.txt


# generate a 2GB data file
cat data-1gb.txt >> data-2gb.txt
cat data-1gb.txt >> data-2gb.txt


# generate a 3GB data file
cat data-1gb.txt >> data-3gb.txt
cat data-2gb.txt >> data-3gb.txt