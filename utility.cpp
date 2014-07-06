/**
 * @brief Kernelized Online Imbalanced Learning with Fixed Buddget
 * Implemented by Junjie Hu
 * Contact: jjhu@cse.cuhk.edu.hk
*/

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <fstream>
#include "utility.h"
//#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
using namespace std;

char *line=NULL;
int max_line_len =0;

char* readline(FILE *input)
{
	int len;	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

void read_matrix(string filename,int** &mat, int &n, int &d)
{
	unsigned int max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename.c_str(),"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		cout<<"can't open input file" << filename<<endl;
		exit(1);
	}

	max_line_len = 1024;
	line = Malloc(char,max_line_len);

	n=d=0;
	bool first=true;
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); 
		while(1)
		{   
			if(first) 
				++d;
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			
		}		
		++n;
		first = false;
	}
	rewind(fp);

	// read data
	mat = Malloc(int*,n);
	i=0;
	while(readline(fp)!=NULL)
	{
		mat[i]=Malloc(int,d);
		j=0;
		char *p = strtok(line," \t"); 
		while(1)
		{   
			mat[i][j++]=atoi(p);
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n')// check '\n' as ' ' may be after the last feature
				break;					
		}		
		++i;
	}
}
