/********************************************************************************
 *	IO.c  -  handle all input/output for lpsd.c				*
 ********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "config.h"
#include "misc.h"
#include "errors.h"
#include "tics.h"
#include "debug.h"
#include "IO.h"
#include "StrParser.h"


static FILE *ifp = 0;			/* input file pointer */
static char curline[DATALEN];		/* currently read input data line */
static double curdata;			/* contains current data */
static double curtime;			/* contains current time */
static double dts;			/* sum of delta t's */
static double dt2s;			/* sum of delta t^2's */
static double *data = 0;		/* pointer to all data */
static int (*read_data) (void);		/* pointer to function reading input data */
static unsigned int timecol;		/* column 1 contains time in s */
static unsigned int colA;		/* read data from column A */
static unsigned int colB;		/* read data from column B */

static void replaceComma(char *s);
static int read_t_A_B(void);
static int read_A_B(void);
static int read_A(void);

/********************************************************************************
 *	replaces commas by decimal dots						*
 *	parameters:								*
 *		char *s		pointer to string to be changed			*
 ********************************************************************************/
static void replaceComma(char *s) {
	unsigned int n;
	
	for (n = 0; s[n] > 0; n++)
		if (s[n] == ',')
			s[n] = '.';

}

/********************************************************************************
 *	reads time and two columns from a file					*
 *	returns:								*
 *		1 on success							*
 *		0 on failure							*
 ********************************************************************************/
static int read_t_A_B(void)
{
	unsigned int n, ok = 0;
	char s[DATALEN];
	char *col;
	double dataA, dataB;
	
	strcpy(&s[0],&curline[0]);
	col=strtok(s,DATADEL);			/* ATTENTION: s gets altered by strtok */
	if (sscanf(col,"%lg",&curtime)==1) 
		ok = 1;

//	printf("time=%f\t",curtime);

	for (n=1; n<colA;n++) col=strtok(NULL,DATADEL);
	if (sscanf(col,"%lg",&dataA)==1) 
		ok = 1; else ok=0;
//	printf("dataA=%f\t",dataA);
	for (n=0; n<colB-colA;n++) col=strtok(NULL,DATADEL);
	if (sscanf(col,"%lg",&dataB)==1) 
		ok = 1; else ok=0;
//	printf("dataB=%f\n",dataB);
	curdata=dataB-dataA;

	return (ok);
}

/********************************************************************************
 *	reads time and one column from a file					*
 *	returns:								*
 *		1 on success							*
 *		0 on failure							*
 ********************************************************************************/
static int read_t_A(void)
{
	unsigned int n, ok = 0;
	char s[DATALEN];
	char *col;
	
	strcpy(&s[0],&curline[0]);

	col=strtok(s,DATADEL);			/* ATTENTION: s gets altered by strtok */
	if (sscanf(col,"%lg",&curtime)==1) 
		ok = 1;

//	printf("time=%f\t",curtime);

	for (n=1; n<colA;n++) col=strtok(NULL,DATADEL);
	if (sscanf(col,"%lg",&curdata)==1) 
		ok = 1; else ok=0;
//	printf("dataA=%f\t",curdata);
	
	return (ok);
}

/********************************************************************************
 *	reads two columns from a file						*
 *	returns:								*
 *		1 on success							*
 *		0 on failure							*
 ********************************************************************************/
static int read_A_B(void)
{
	unsigned int n, ok = 0;
	char s[DATALEN];
	char *col;
	double dataA, dataB;

	strcpy(&s[0],&curline[0]);
	
	col=strtok(s,DATADEL);			/* ATTENTION: s gets altered by strtok */
	for (n=1; n<colA;n++) col=strtok(NULL,DATADEL);
	if (sscanf(col,"%lg",&dataA)==1) 
		ok = 1; else ok=0;
//	printf("dataA=%f\t",dataA);
	for (n=0; n<colB-colA;n++) col=strtok(NULL,DATADEL);
	if (sscanf(col,"%lg",&dataB)==1) 
		ok = 1; else ok=0;
//	printf("dataB=%f\n",dataB);
	curdata=dataB-dataA;

	return (ok);
}

/********************************************************************************
 *	reads one column from a file						*
 *	returns:								*
 *		1 on success							*
 *		0 on failure							*
 ********************************************************************************/
static int read_A(void)
{
	unsigned int n, ok = 0;
	char s[DATALEN];
	char *col;

	strcpy(&s[0],&curline[0]);
	
	col=strtok(s,DATADEL);			/* ATTENTION: s gets altered by strtok */
	for (n=1; n<colA;n++) col=strtok(NULL,DATADEL);
	if (sscanf(col,"%lg",&curdata)==1) 
		ok = 1; else ok=0;
//	printf("dataA=%f\n",curdata);

	return (ok);
}

/********************************************************************************
 *	reads one line from a file					*
 *	parameters:								*
 *		int comma		1 if comma is decimal point, 0 otherwise*
 *	returns:								*
 *		2 if first character is comment character #			*
 *		1 on success							*
 *		0 on failure							*
 *		copies line of data into curline				*
 ********************************************************************************/
static int read_lof(int comma) {
	int ok=0;
	
	if (0 != fgets(&curline[0], DATALEN, ifp)) {		/* read max. DATALEN-1 characters */
		ok=1;
		/* replace commas by decimal points */
		if (comma == 1) replaceComma(curline);
		/* test for comment character */
		if (curline[0]=='#') ok=2;
	}
	return (ok);

}

/* returns 1 if file fn exists, 0 otherwise */
int exists(char *fn)
{
	int ok = 0;
	ifp = fopen(fn, "r");
	if (ifp != 0) {
		ok = 1;
		fclose(ifp);
	}
	return (ok);
}


/********************************************************************************
 *	reads file *fn, counts number of data points and determines mean of data
 *										
 *	Parameters
 *		t	1 first column contains time in s
 *		A	number of column to process
 *		B	number of column to process in combination
 ********************************************************************************
 	Naming convention	source code	publication
				fs		f_s
				ndata		N
 ********************************************************************************/
void probe_file(unsigned int t, unsigned int A, unsigned int B)
{
	timecol=t;
	colA=A,
	colB=B;

	/* select reading routine */
	if ((timecol==1) & (colB>0)) read_data=read_t_A_B;
	else if ((timecol==1) & (colB==0)) read_data=read_t_A;
	else if ((timecol==0) & (colB>0)) read_data=read_A_B;
	else if ((timecol==0) & (colB==0)) read_data=read_A;

	if (read_data==NULL) gerror("No file reading routine selected!\n");
}

/*
	writes a comment line to *ofp with information on what quantity will be saved in what column
	
	input: 	ofp	output file name
		gt	gnuplot terminal
*/
static void writeHeaderLine(FILE *ofp, tGNUTERM * gt) {
	unsigned int c;
	char tmp[SLEN];
	
	fprintf(ofp,"# ");;
	for (c = 0; c < strlen((*gt).fmt); c++) {
		switch ((*gt).fmt[c]) {
		case 'f':
			fprintf(ofp, "Frequency (Hz)	");
			break;
		case 'd':
			fprintf(ofp, "LSD	");
			break;
		case 'D':
			fprintf(ofp, "PSD	");
			break;
		case 's':
			fprintf(ofp, "LS	");
			break;
		case 'S':
			fprintf(ofp, "PS	");
			break;
		case 'N':
			fprintf(ofp, "AVG	");
			break;
		case 'u':
			fprintf(ofp, "STD LSD	");
			break;
		case 'U':
			fprintf(ofp, "VAR PSD	");
			break;
		case 'v':
			fprintf(ofp, "STD LS	");
			break;
		case 'V':
			fprintf(ofp, "VAR PS	");
			break;
		case 'R':
			fprintf(ofp, "RBW (Hz)	");
			break;
		case 'b':
			fprintf(ofp, "Bin	");
			break;
		case 'r':
			fprintf(ofp, "Re(PSD)	");
			break;
		case 'i':
			fprintf(ofp, "Im(PSD)	");
                        break;
		default:
			strcpy(&tmp[0],&((*gt).fmt[0]));
			(*gt).fmt[c+1]=0;
			message1("\tWARNING: Identifier %s in gnuplot terminal not recognized",&((*gt).fmt[c]));
			strcpy(&((*gt).fmt[0]),&tmp[0]);
			break;
		}
	}
	fprintf(ofp,"\n");;
}

/*
	writes the output data that lpsd calculated to file
	
	input:	ofp	input file pointer
		cfg	configuration information for nspec
		data	actual data to write
		gt	gnuplot terminal - what to write
*/
static void writeData(FILE *ofp, tCFG * cfg, tDATA * data, tGNUTERM * gt) {
	int i;
	unsigned int c;
	
	for (i = 0; i < (*cfg).nspec; i++) {
		for (c = 0; c < strlen((*gt).fmt); c++) {
			switch ((*gt).fmt[c]) {
			case 'f':
				fprintf(ofp, "%.10e\t", (*data).fspec[i]);
				break;
			case 'd':
				fprintf(ofp, "%.10e\t", sqrt((*data).psd[i]));
				break;
			case 'D':
				fprintf(ofp, "%.10e\t", (*data).psd[i]);
				break;
			case 's':
				fprintf(ofp, "%.10e\t", sqrt((*data).ps[i]));
				break;
			case 'S':
				fprintf(ofp, "%.10e\t", (*data).ps[i]);
				break;
			case 'N':
				fprintf(ofp, "%d\t", (*data).avg[i]);
				break;
			case 'u':
				fprintf(ofp, "%.10e\t", sqrt((*data).varpsd[i]));
				break;
			case 'U':
				fprintf(ofp, "%.10e\t", (*data).varpsd[i]);
				break;
			case 'v':
				fprintf(ofp, "%.10e\t", sqrt((*data).varps[i]));
				break;
			case 'V':
				fprintf(ofp, "%.10e\t", (*data).varps[i]);
				break;
			case 'R':
				fprintf(ofp, "%.10e\t",
					(*cfg).fsamp / (double) (*data).nffts[i]);
				break;
			case 'b':
				fprintf(ofp, "%.10e\t", (*data).bins[i]);
				break;
			case 'r':
				fprintf(ofp, "%.10e\t", (*data).psd_real[i]);
				break;
			case 'i':
				fprintf(ofp, "%.10e\t", (*data).psd_imag[i]);
				break;
			default:
				break;
			}
		}
		fprintf(ofp, "\n");
	}
}


/*
	prints the command line the program was called with to a string
	
	inputs:	dst	pointer to write result to
			must point to variable that has enough space to write data to
		argc	number of command line arguments
		argv	argument strings
*/
void printCommandLine(char *dst, int argc, char *argv[]) {
	int i,b;
	
	b=0;
	for (i=0;i<argc;i++) {
		sprintf(&dst[b],"%s ",argv[i]);
		b=strlen(dst);
	}
			
}

/*
	write information to string: 
*/
static void writeComment(char *dest, tCFG *cfg, tWinInfo *wi, tGNUTERM *gt, tDATA *data, int argc, char *argv[]) {
	char *stm;		/* string with time and date */
	time_t tm;		/* time in epoch */
	char c[CLEN];
	char cmdline[CMTLEN];
	
	/* date, time, and lpsd version */
	tm = time(NULL);
	stm = ctime(&tm);
	sprintf(&dest[0], "# output from %s, generated ", LPSD_VERSION);
	strcat(dest, stm);
	
	/* command line */
	printCommandLine(&cmdline[0],argc, argv);
	sprintf(&dest[strlen(dest)],"# Command line: %s\n#",cmdline);

	/* info on files, window, data, output */
	printConfig(&c[0],*cfg, *wi, *gt, *data);
	rplStr(&c[0], "\n", "\n# ");
	strcat(dest,c);
	sprintf(&dest[strlen(dest)],"\n");
}

/*
	# identifiers for output file format
	#
	# f	frequency
	# d	linear spectral density
	# D	power spectral density
	# s	linear spectrum
	# S	power spectrum
	# N	number of averages
	# u	standard deviation of linear spectral density
	# U	variance of power spectral density
	# v	standard deviation of linear spectrum
	# V	variance of power spectrum
	# R	resolution bandwidth
	# b	bin number

	The format of the output file is stored in gt[gti].fmt

*/
void writeOutputFile(tCFG * cfg, tDATA * data, tGNUTERM * gt, tWinInfo *wi, int argc, char *argv[]) {
	FILE *ofp;
	char cmt[CMTLEN];

	ofp = fopen((*cfg).ofn, "w");
	if (0 == ofp)
		gerror1("Error opening output file.. Aborting.", (*cfg).ofn);

	writeComment(&cmt[0], cfg, wi, gt, data, argc, argv);
	fprintf(ofp,"%s",cmt);

	writeHeaderLine(ofp, gt);
	writeData(ofp, cfg, data, gt);
	
	fclose(ofp);
}

static double getLSD(tDATA *data, int i) {
	return sqrt((*data).psd[i]);
}

static double getLS(tDATA *data, int i) {
	return sqrt((*data).ps[i]);
}

static double getPSD(tDATA *data, int i) {
	return (*data).psd[i];
}

static double getPS(tDATA *data, int i) {
	return (*data).ps[i];
}

/*
	scan gt.fmt string and determine calibration of first y column
	scan data and return minimum and maximum value for first y column
*/
static void getSpan(tCFG *cfg, tDATA *data, tGNUTERM *gt, double *ymin, double *ymax) {
	int i;
	int c;
	double (*getData)(tDATA *data, int i)=getLSD;
	
	/* scan gt.fmt string and determine calibration of first y column */
	for (c = strlen((*gt).fmt)-1; c>=0; c--) {
		switch ((*gt).fmt[c]) {
		case 'd':				/* LSD */
			getData = getLSD;
			break;
		case 'D':				/* PSD */
			getData = getPSD;
			break;
		case 's':				/* LS */
			getData = getLS;
			break;
		case 'S':				/* PS*/
			getData = getPS;
			break;
		default:
			break;
		}
	}
	
	/* scan data and return minimum and maximum value for first y column */
	*ymin=1e300; *ymax=-1e300;
	for (i=0;i<(*cfg).nspec; i++) {
		if (getData(data, i)<*ymin) *ymin=getData(data, i);
		if (getData(data, i)>*ymax) *ymax=getData(data, i);
	}
}

/*
	write general information
	parse and write gnuplot commands

	%x xtics determined by lpsd
	%y ytics determined by lpsd
	%f input data file
	%g input file name without extension
	%o output file name
	%p output file name without extension
	%s paramater string from lpsd command line

*/
void writeGnuplotFile(tCFG *cfg, tDATA *data, tGNUTERM *gt, tWinInfo *wi, int argc, char *argv[]) {
	FILE *gfp;
	char cmt[CMTLEN];
	char ifnb[FNLEN];
	char ofnb[FNLEN];
	char xtics[TICLEN];
	char ytics[TICLEN];
	double ymin, ymax;
	
	gfp = fopen((*cfg).gfn, "w");
	if (0 == gfp)
		gerror1("Error opening", (*cfg).gfn);
	
	/* write general information */
	writeComment(&cmt[0], cfg, wi, gt, data, argc, argv);
	fprintf(gfp,"%s",cmt);

	/* replace %x, %y, %f %g, %o, %p, %s in gnuplot command string */
	basename((*cfg).ifn, ifnb);			/* copy input file basename to ifnb */
	basename((*cfg).ofn, ofnb);			/* copy output file basename to ofnb */
	rplStr(&(*gt).cmds[0],"%f",(*cfg).ifn);
	rplStr(&(*gt).cmds[0],"%g",ifnb);
	rplStr(&(*gt).cmds[0],"%s",(*cfg).param);
	rplStr(&(*gt).cmds[0],"%o",(*cfg).ofn);
	rplStr(&(*gt).cmds[0],"%p",ofnb);

	maketics(xtics, 'x', (int) floor(log10((*cfg).fmin)) - 1, ceil((int) log10((*cfg).fmax)) + 1);
	getSpan(cfg, data, gt, &ymin, &ymax);
	maketics(ytics, 'y', (int) floor(log10(ymin)) - 1, ceil((int) log10(ymax)) + 1);
	rplStr(&(*gt).cmds[0],"%x",xtics);
	rplStr(&(*gt).cmds[0],"%y",ytics);
	/* write gnuplot commands */
	fprintf(gfp,"%s",(*gt).cmds);
	
	fclose(gfp);
}

void saveResult(tCFG * cfg, tDATA * data, tGNUTERM * gt, tWinInfo *wi, int argc, char *argv[])
{
	double ymin, ymax;
	int k;
//	FILE * file1;

	/* Load values from tempory backup into their respective columns */
//	printf("mhhhhh\n");
//	file1 = fopen(cfg->ofn, "r");
//	printf("mh\n");
//	for(k = 0; k < cfg->nspec; k++){
////	    fscanf(file1, "%lf %lf %d", &(*data).psd[k], &(*data).ps[k], &(*data).avg[k]);
////        printf("\t%e, %e, %d\n", data->psd[k], data->ps[k], data->avg[k]);
//	    fscanf(file1, "%lf %lf %d", &(data->psd[k]), &(data->ps[k]), &(data->avg[k]));
//	}
//	printf("mh\n");
//	fclose(file1);
//	printf("mh\n");
	
	/* write output file with colums specified in gt */
	writeOutputFile(cfg, data, gt, wi, argc, argv);

	/* find minimum and maximum of first y column */
	getSpan(cfg, data, gt, &ymin, &ymax);
	
	/* write gnuplot file */
	writeGnuplotFile(cfg, data, gt, wi, argc, argv);
}

// @brief Read the contents of a metadata and return pointer to hdf5_contents struct.
// @brief This include ids of file, dataset, dataspace, as well as rank/dims info.
void read_hdf5_file(struct hdf5_contents *contents,
                    char *filename, char *dataset_name)
{
    // Open data
    hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dataset = H5Dopen(file, dataset_name, H5P_DEFAULT);
    hid_t dataspace = H5Dget_space(dataset);

    // Get dims
    hsize_t rank = H5Sget_simple_extent_ndims(dataspace);
    hsize_t dims[rank];
    herr_t status = H5Sget_simple_extent_dims(dataspace, dims, NULL);

    // Save info to struct
    contents->file = file;
    contents->dataset = dataset;
    contents->dataspace = dataspace;
    contents->rank = rank;
    contents->dims = dims;
}


// @brief Open a new HDF5 file and create a dataspace/dataset inside (of type double)
void open_hdf5_file(struct hdf5_contents *contents,
                    char *filename, char *dataset_name,
                    hsize_t rank, hsize_t *dims) {
    // Create file in truncation mode
    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Create dataspace and dataset in file
    hid_t dataspace = H5Screate_simple(rank, dims, NULL);
    hid_t dataset = H5Dcreate(file, dataset_name, H5T_NATIVE_DOUBLE, dataspace,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Save info to struct
    contents->file = file;
    contents->dataset = dataset;
    contents->dataspace = dataspace;
    contents->rank = rank;
    contents->dims = dims;
}


// @brief Write a data matrix to an existing HDF5 file with an existing dataspace
// @param offset, count: specify where in the file dataspace to write the data
void write_to_hdf5(struct hdf5_contents *_contents, double *data,
                   hsize_t *offset, hsize_t *count,
                   hsize_t data_rank, hsize_t *data_count) {
    // Select hyperslab in file dataspace
    // Keep in mind: offset/count need as many dimensions as contents->rank
    herr_t status = H5Sselect_hyperslab(_contents->dataspace, H5S_SELECT_SET,
                                        offset, NULL, count, NULL);

    // Create memory dataspace
    hid_t memspace = H5Screate_simple(data_rank, data_count, NULL);
    hsize_t data_offset[data_rank];
    for (int i = 0; i < (int) data_rank; i++) data_offset[i] = 0;
    status = H5Sselect_hyperslab(memspace, H5S_SELECT_SET,
                                 data_offset, NULL, data_count, NULL);

    // Write data to dataset
    status = H5Dwrite(_contents->dataset, H5T_NATIVE_DOUBLE,
                      memspace, _contents->dataspace, H5P_DEFAULT, data);

    // Clean-up
    H5Sclose(memspace);
    status = H5Sselect_none(_contents->dataspace);
}

// Wrapper for read_from_dataset_stride with no stride
void read_from_dataset(struct hdf5_contents *contents, hsize_t *offset,
                       hsize_t *count, hsize_t data_rank, hsize_t *data_count,
                       double *data_out)
{
    int rank = (int)contents->rank;
    hsize_t stride[rank];
    for (int i = 0; i < rank; i++) stride[i] = 1;
    read_from_dataset_stride(contents, offset, count, stride, data_rank, data_count, data_out);
}


// @brief Fill data_out with contents from the hdf5 dataset, according to offset/count
// @param data_out: should have the same dimensions as count
// @param count: size of dataset to read
// @param offset: position in the dataset at which to start reading
// @param stride: stride parameter when creating hyperslab
void read_from_dataset_stride(struct hdf5_contents *contents, hsize_t *offset,
                              hsize_t *count, hsize_t *stride,
                              hsize_t data_rank, hsize_t *data_count,
                              double *data_out)
{
    // Use hyperslab to read partial file contents out
    herr_t status = H5Sselect_hyperslab(contents->dataspace, H5S_SELECT_SET,
                                        offset, stride, count, NULL);
    hid_t memspace = H5Screate_simple(data_rank, data_count, NULL);

    status = H5Dread(contents->dataset, H5T_NATIVE_DOUBLE, memspace,
                     contents->dataspace, H5P_DEFAULT, data_out);
    // Clean up
    H5Sclose(memspace);
    status = H5Sselect_none(contents->dataspace);
}

void close_hdf5_contents(struct hdf5_contents *contents)
{
    // TODO: add if statements (only needed if close_hdf5_contents may be called in different circumstances)
    H5Dclose(contents->dataset);
    H5Sclose(contents->dataspace);
    H5Fclose(contents->file);
}
