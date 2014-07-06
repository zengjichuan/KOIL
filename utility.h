/**
 * @brief Kernelized Online Imbalanced Learning with Fixed Buddget
 * Implemented by Junjie Hu
 * Contact: jjhu@cse.cuhk.edu.hk
*/

#ifndef _UTILITY_H
#define _UTILITY_H

#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <cmath>
#include <fstream>
//#include "svm.h"
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
using namespace std;

extern char *line;
extern int max_line_len;

// clone src to dst
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}

// copy the [head tail] entries from the c-th column of src to dst
template <class S> static inline void get_column(S*& dst, S** src,int head, int tail, int c)
{
	dst = Malloc(S,tail-head+1);
	if(head<0||src[tail][c]==NULL){
        cout<<"head "<<head<<" or tail "<<tail<<" out of range."<<endl;
		exit(1);
	}
	for(int i=head;i<=tail;i++)
		dst[i-head]=src[i][c];
}


// copy the [head tail] entries from the c-th column of src to dst
template <class S> static inline void get_column(S*& dst, S** src,int head, int tail, int n,  int c)
{
    dst = Malloc(S,tail-head+1);
    if(head<0||tail>=n){
        cout<<"head "<<head<<" or tail "<<tail<<" out of range (0,"<<n<<")."<<endl;
        exit(1);
    }
    for(int i=head;i<=tail;i++)
        dst[i-head]=src[i][c];
}
// read each line from the file
char* readline(FILE *input);

// read integeter matrix from file
void read_matrix(string filename,int** &mat, int &n, int &d);


template <typename T>
void ordered(std::vector<T> const& values, std::vector<int>& indices) {
    indices.assign(values.size(),0);
    std::iota(begin(indices), end(indices), static_cast<size_t>(0));

    std::sort(
        indices.begin(),indices.end(),
        [&](size_t a, size_t b) { return values[a] >= values[b]; }
    );
}

template<class T>
bool comparator ( const std::pair<T,int>& l, const std::pair<T,int>& r)
{ return l.first > r.first; }

template<class T>
void write_log(T* data, int n,string note, string filename)
{
    std::ofstream of(filename,ios::app);
    of<<note<<std::endl;
    for(int i =0;i<n;i++){
        of<<data[i]<<" ";
    }
    of<<std::endl;
    of.close();
}


template<class T>
void write_matrix(T** data, int n, int d, string filename)
{
    std::ofstream of(filename,ios::app);
//    of<<note<<std::endl;
    for(int i =0;i<n;i++){
        for(int j=0;j<d;j++){
            of<<data[i][j]<<" ";
        }of<<std::endl;
    }
//    of<<std::endl;
    of.close();
}


#endif 
