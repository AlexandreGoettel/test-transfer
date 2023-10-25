#!/usr/bin/env sh

# Declare data file
data_filename=$filename
dataset=strain

# Number of seconds of data
TSlength=$TSlength

# Start and end frequencies
fmin=$fmin
fmax=$fmax

# Sampling rate
TSfreq=$fsamp

# Number of frequency bins
Nfreqs_full=$Jdes

# Iteration parameters (set iter=0, batch_size=${Nfreqs_full} for h=1)
iter=0
batch_size=${Nfreqs_full}

# Output file
outfilename=$output_filename

$LPSD_PATH \
        -A 2 \
        -b 0 \
        -e ${TSlength} \
        -f ${TSfreq} \
        -h 1 \
        -i ${data_filename}\
        -D ${dataset}\
        -l 20 \
        -n ${batch_size} \
        -o ${outfilename} \
        -r 0 \
        -s ${fmin} \
        -t ${fmax} \
        -T \
        -w -2 \
        -p 238.13 \
        -x 1 \
        -N ${iter} \
        -J ${Nfreqs_full} \
	-M 30\
	-E 10
