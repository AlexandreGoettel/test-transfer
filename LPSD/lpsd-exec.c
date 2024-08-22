/********************************************************************************
    lpsd-exec.c
			  
    2003, 2004, 2006 
    by Michael Troebs, mt@lzh.de and Gerhard Heinzel, ghh@mpq.mpg.de

    calculate spectra from time series using discrete Fourier 
    transforms at frequencies equally spaced on a logarithmic axis

 ********************************************************************************/
#define SINCOS

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>		/* gettimeofday, timeval */
#include <assert.h>
#include "hdf5.h"

#include "config.h"
#include "ask.h"
#include "IO.h"
#include "genwin.h"
#include "debug.h"
#include "lpsd.h"
#include "misc.h"
#include "ArgParser.h"
#include "StrParser.h"
#include "lpsd-exec.h"
#include "errors.h"

extern double round(double x);
/*
gcc (GCC) 3.3.1 (SuSE Linux) gives warning: implicit declaration of function `round'
without this line - math.h defines function round, but its declaration seems to be
missing in math.h 
*/

/********************************************************************************
 * 	global variables						   	
 ********************************************************************************/

/* Program documentation. */
static char doc[] = "\nlpsd - a program to calculate spectral estimates by Michael Troebs\
  \nand Gerhard Heinzel, (c) 2003, 2004, 2006\n\n";

tCFG cfg;			/* configuration data */
tGNUTERM gt;			/* gnuplot terminal in use */
tDATA data;			/* info on input data and results */
tWinInfo wi;			/* info on window function */

/********************************************************************************
 * 	functions								
 ********************************************************************************/

/********************************************************************************
 *	read the user's input from the keyboard
 ********************************************************************************/
void getUserInput()
{
	tGNUTERM agt[MAXGNUTERM];		/* all gnuplot terminals */
	double maxt;				/* maximum stop time */
	double fsamp;
	double xov;
	double rov;
	int i;

	if (cfg.askifn == 1)
		asks("Input file", cfg.ifn);
	if (!exists(cfg.ifn))
		gerror("Input file name does not exist");

	if (cfg.asktime == 1)
		aski("Time in column 1 (0 : no, 1 : yes)?", &cfg.time);
	if (cfg.askcolA == 1)
		aski("Number of column to process", &cfg.colA);
	
	if ((cfg.askcolB>0) | (cfg.colB>0)) {
		do {
			if ((cfg.askcolB == 1) | (cfg.colB<cfg.colA)) {
				if (cfg.colB<cfg.colA) printf("This column number must be larger than the previous one or zero!\n");
				aski("Process difference to column (0 if none)", &cfg.colB);
			}	
		} while ((cfg.colB!=0) & (cfg.colB<cfg.colA));
	}

	/* create output filename based on input file name, parameter */
	parse_fgsC(cfg.ofn,cfg.ifn,cfg.param,cfg.colA,cfg.colB);
	if (cfg.askofn == 1)
		asks("Output file", cfg.ofn);
	/* create gnuplot filename based on input file name, parameter */
	parse_fgsC(cfg.gfn,cfg.ifn,cfg.param,cfg.colA,cfg.colB);
	/* create gnuplot filename based on output file name */
	parse_op(cfg.gfn,cfg.ofn);
	if (cfg.askgfn == 1)
		asks("Gnuplot file", cfg.gfn);

	data.mean = 0;  // Strange but ok..
	probe_file(cfg.time, cfg.colA, cfg.colB);
	/*
		if time is contained in first column and sampling frequency is not given on command line
		then raise error.
	*/
	if ((cfg.time==1) && (cfg.cmdfsamp==0)) gerror("No sampling frequency given!");

	if (cfg.askfsamp == 1)
		askd("Sampling frequency (Hz)", &cfg.fsamp);
	if (cfg.tmax < 0)
		cfg.tmax = (double) (data.ndata - 1) / (double) cfg.fsamp;
	maxt = cfg.tmax;
	do {
		if (cfg.asktmin == 1)
			askd("Start time for spectrum estimation (s)", &cfg.tmin);
		if ((cfg.tmin < 0) || (cfg.tmin > maxt))
			printf("  Please enter a value between 0 and %g\n", maxt);
	}
	while ((cfg.tmin < 0) || (cfg.tmin > maxt));
	do {
		if (cfg.asktmax == 1)
			askd("Stop time for spectrum estimation (s)", &cfg.tmax);
		if ((cfg.tmax < 0) || (cfg.tmax > maxt))
			printf("  Please enter a value between 0 and %g\n", maxt);
	}
	while ((cfg.tmax < 0) || (cfg.tmax > maxt));
	data.nread = floor((cfg.tmax - cfg.tmin) * cfg.fsamp + 1);
	if (cfg.askWT == 1)
		aski("Window type -2 = Kaiser window, -1 = flat-top, 0..30 window function by number", &cfg.WT);
	if ((cfg.WT == -2) || (cfg.WT == -1))
		if (cfg.askreqPSLL == 1)
			askd("requested PSLL - peak side lobe level", &cfg.reqPSLL);
	set_window(cfg.WT, cfg.reqPSLL, &wi.name[0], &wi.psll, &rov,
		   &wi.nenbw, &wi.w3db, &wi.flatness, &wi.sbin);
	if (cfg.sbin<0) cfg.sbin=wi.sbin;
	
	if (cfg.askovlp == 1) {
		askd("Overlap in %",&cfg.ovlp);
	}
	/* 
		if ovlp was not given on command line and automatic overlap calculation was selected, 
		then use recommended overlap from set_window
	*/
	printf("cfg.cmdovlp=%d, cfg.ovlp=%f\n",cfg.cmdovlp,cfg.ovlp);
	if ((cfg.cmdovlp==0) && (cfg.ovlp<0)) cfg.ovlp=rov;
	
	if (cfg.askMETHOD == 1)
		aski("METHOD for frequency nodes calculation (0 or 1)", &cfg.METHOD);
	
	if (cfg.fmin < 0) {
		xov = (1. - cfg.ovlp / 100.);
		cfg.fmin = cfg.sbin / (data.nread/cfg.fsamp) * (1 + xov * (cfg.minAVG - 1));
	}
	if (cfg.fres < 0) {
		xov = (1. - cfg.ovlp / 100.);
		cfg.fres = 1. / (data.nread/cfg.fsamp) * (1 + xov * (cfg.minAVG - 1));
	}
	if (cfg.fmax < 0)
		cfg.fmax = cfg.fsamp / 2.0;
	if (cfg.askfmin == 1)
		askd("Min. frequency", &cfg.fmin);
	if (cfg.asksbin == 1)
		askd("Min. freq. bin", &cfg.sbin);
	if (cfg.askfmax == 1)
		askd("Max. frequency", &cfg.fmax);
	if (cfg.METHOD == 0 || cfg.METHOD == 1) {	
		if (cfg.asknspec == 1)
			askil("Number of samples in spectrum", &cfg.nspec);
		if (cfg.askminAVG == 1)
			aski("Minimum number of averages", &cfg.minAVG);	
		if (cfg.askdesAVG == 1)
			aski("Desired number of averages", &cfg.desAVG);
	}
	if (cfg.askulsb == 1)
		askd("Scaling factor", &cfg.ulsb);
	if ((cfg.ngnuterm > 0) && (cfg.askgt==1)) {
		printf("Gnuplot terminals:\n");
		for (i = 0; i < cfg.ngnuterm; i++) {
			getGNUTERM(i,&agt[i]);
			printf("\t(%d) %s\n",i,agt[i].identifier);
		}
		aski("Gnuplot terminal", &cfg.gt);
	}
}

void getDefaultValues()
{
	double maxt;				/* maximum stop time */
	double fsamp;
	double xov;
	double rov;
	
  	printf("Using defaults...\n");

	if (!exists(cfg.ifn))
	    gerror("input file name does not exist");
	/* handle output filename */
	parse_fgsC(cfg.ofn,cfg.ifn,cfg.param,cfg.colA,cfg.colB);
	parse_fgsC(cfg.gfn,cfg.ifn,cfg.param,cfg.colA,cfg.colB);
	parse_op(cfg.gfn,cfg.ofn);
    data.mean = 0;  // Strange but ok
    probe_file(cfg.time, cfg.colA, cfg.colB);
	/*
		if time is contained in first column and sampling frequency is not given on command line
		then raise error.
	*/
	if ((cfg.time==1) && (cfg.cmdfsamp==0)) gerror("No sampling frequency given!");

	if (cfg.tmax < 0) {
	    cfg.tmax = (double) (data.ndata - 1) / (double) cfg.fsamp;
	}    
	maxt = cfg.tmax;
	data.nread = floor((cfg.tmax - cfg.tmin) * cfg.fsamp + 1);
	set_window(cfg.WT, cfg.reqPSLL, &wi.name[0], &wi.psll, &rov,
		   &wi.nenbw, &wi.w3db, &wi.flatness, &wi.sbin);
	if ((cfg.cmdovlp==1) && (cfg.ovlp<0))
		cfg.ovlp=rov;
	if (cfg.sbin<0)
		cfg.sbin=wi.sbin;
	if (cfg.fmin < 0) {
		xov = (1. - cfg.ovlp / 100.);
		cfg.fmin = cfg.sbin / (data.nread/cfg.fsamp) * (1 + xov * (cfg.minAVG - 1));
	}
	if (cfg.fres < 0) {
		xov = (1. - cfg.ovlp / 100.);
		cfg.fres = 1. / (cfg.tmax - cfg.tmin) * (1 + xov * (cfg.minAVG - 1));
	}    
	if (cfg.fmax < 0)
		cfg.fmax = cfg.fsamp / 2.0;
}

void memalloc(tCFG *cfg, tDATA *data, DataPointers *dp) {
	// Adjust the following arrays following tData
    double **doubleMembers[] = {&data->fspec, &data->bins, &data->ps, &data->psd, &data->varps, &data->varpsd,
                                &data->psd_real, &data->psd_imag, &data->psd_raw, &data->psd_raw_real, &data->psd_raw_imag};
    int **intMembers[] = {&data->avg};
    long int **liMembers[] = {&data->nffts};

    dp->numDoublePointers = sizeof(doubleMembers) / sizeof(doubleMembers[0]);
    dp->doublePointers = (double **) xmalloc(dp->numDoublePointers * sizeof(double *));
    dp->numIntPointers = sizeof(intMembers) / sizeof(intMembers[0]);
    dp->intPointers = (int **) xmalloc(dp->numIntPointers * sizeof(int *));
    dp->numLiPointers = sizeof(liMembers) / sizeof(liMembers[0]);
    dp->liPointers = (long int **) xmalloc(dp->numLiPointers * sizeof(long int *));

    for (int i = 0; i < dp->numDoublePointers; ++i) {
        *(doubleMembers[i]) = (double *) xmalloc(cfg->nspec * sizeof(double));
        dp->doublePointers[i] = *(doubleMembers[i]);
    }

    for (int i = 0; i < dp->numIntPointers; i++) {
		*(intMembers[i]) = (int *) xmalloc(cfg->nspec * sizeof(int));
		dp->intPointers[i] = *(intMembers[i]);
    }
    for (int i = 0; i < dp->numLiPointers; i++) {
		*(liMembers[i]) = (long int *) xmalloc(cfg->nspec * sizeof(long int));
		dp->liPointers[i] = *(liMembers[i]);
    }
}

void memfree(DataPointers *dp) {
    for (int i = 0; i < dp->numDoublePointers; ++i)
        xfree(dp->doublePointers[i]);
    xfree(dp->doublePointers);
    for (int i = 0; i < dp->numIntPointers; i++)
        xfree(dp->intPointers[i]);
    xfree(dp->intPointers);
}

void checkParams() {
	double xov, fm;
	
	xov = (1. - cfg.ovlp / 100.);
	fm = cfg.sbin / (data.nread/cfg.fsamp) * (1 + xov * (cfg.minAVG - 1));

	if (cfg.fmax>cfg.fsamp/2.0)
		gerror("Largest frequency cannot be bigger than fsamp/2!");
	if ((cfg.fmin*(1.+1e-6))<fm) {
		printf("min. req. freq:\t%.2e, min. poss. freq:\t%.2e\n",cfg.fmin,fm);
		gerror("Reduce minAVG or increase minimum frequency!");
	}
	if (cfg.METHOD==0) {
		if (cfg.cmdfres) message("frequency resolution parameter is ignored in LPSD mode!");
	}
}

/********************************************************************************
 * 	main								   	
 ********************************************************************************/

int main(int argc, char *argv[])
{
	char s[CLEN];
	/* Wipe previous memory somehow? XhereX */	
	readConfigFile();
	getConfig(&cfg);
	printf("%s",doc);
	parseArgs(argc, argv, &cfg);
	if (cfg.usedefs==0) getUserInput();
	else getDefaultValues();
	getGNUTERM(cfg.gt, &gt);
	printConfig(&s[0], cfg, wi, gt, data);
	printf("%s",s);

	checkParams();
	DataPointers dp;
	memalloc(&cfg, &data, &dp);

	calculateSpectrum(&cfg, &data);
	saveResult(&cfg, &data, &gt, &wi, argc, argv);

	memfree(&dp);

	return EXIT_SUCCESS;
}
