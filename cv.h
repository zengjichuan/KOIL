#ifndef CV_H
#define CV_H
#include "svm.h"
#include "KOIL.h"
#include "utility.h"
#include <string>
#include <vector>
using namespace std;

void get_cv_list(KOIL& koil,int* &id_train,double* &y_train,int* &idx_j,int& n_train)
{
    svm_problem& prob = koil.prob;

    n_train =0;
    int head = 0;
    int n_test=0;
    for(int t=0;t<prob.n_cv;t++){
        if(prob.idx_cv[t][0]!=1)
            n_train++;
        else n_test++;
    }
    if(n_train+n_test!=prob.n || prob.n!= prob.n_cv){
        cout<<"n_train+n_test!= prob.n or prob.n!= prob.n_cv"<<endl;
        exit(1);
    }

    id_train = Malloc(int,n_train);
    y_train = Malloc(double,n_train);

    // get id_train, id_test, y_train, y_test
    for(int t=0,cnt_tr=0,cnt_te=0;t<prob.n_cv;t++){
        if(prob.idx_cv[t][0]!=1){
            id_train[cnt_tr]  = t;
            y_train[cnt_tr++] = prob.y[t];
        }
    }

    // search the best parameters by cross validation
    int tail = head+n_train-1;
    get_column(idx_j, prob.idx_Asso, head,tail,prob.n_Asso,0);
}



#endif // CV_H
