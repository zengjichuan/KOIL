/**
 * @brief Kernelized Online Imbalanced Learning with Fixed Buddget
 * Implemented by Junjie Hu
 * Contact: jjhu@cse.cuhk.edu.hk
*/

#include "svm.h"
#include "KOIL.h"
#include "utility.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <mutex>
using std::vector;
using std::cout;
using std::endl;
using boost::mutex;
using std::pow;

double C;
double g;
string losstype;

void online(KOIL& koil)
{
    svm_problem prob=koil.prob;
    koil_result rs_result;
    koil_result fifo_result;
    rs_result.initial_result(20);
    fifo_result.initial_result(20);

    auto run_rs_fifo = [&](int* id_train, int n_train, int* id_test, int n_test, int k)
    {
        //2. KOIL_FIFO++
        svm_model fifo_model;
        fifo_model.initialize(100);
        fifo_model.param.C = C;
        fifo_model.param.gamma = g;
        koil.fifo_plus(id_train, n_train, id_test, n_test, losstype, fifo_model,
                  fifo_result.auc[k-1],fifo_result.accuracy[k-1],
                fifo_result.time[k-1],fifo_result.err_cnt[k-1]);
        cout<<"FIFO++"<<endl<<"auc="<<fifo_result.auc[k-1]<<endl;
        cout<<"accuracy="<<fifo_result.accuracy[k-1]<<endl;

        //1. KOIL_RS++
        svm_model rs_model;
        rs_model.initialize(100);
        rs_model.param.C = C;
        rs_model.param.gamma = g;
        koil.rs_plus(id_train, n_train, id_test, n_test, losstype, rs_model,
                rs_result.auc[k-1],rs_result.accuracy[k-1],
                rs_result.time[k-1],rs_result.err_cnt[k-1]);
        cout<<"RS++"<<endl<<"auc="<<rs_result.auc[k-1]<<endl;
        cout<<"accuracy="<<rs_result.accuracy[k-1]<<endl;


    };

    //std::vector<boost::shared_ptr<boost::thread>> thread_pool(1);
    std::vector<boost::shared_ptr<boost::thread>> thread_pool(20);
    //for(int i = 1; i<=1; i++){
    for(int i=1;i<=4;i++){
        //int* & indice = prob.idx_cv[:,i-1];
        int head = 0;
        //for(int j=1;j<=1;j++){
        for(int j=1;j<=5;j++){
            int k=5*(i-1)+j;
            cout<<"The"<<k<<"-th trial"<<endl;

            int n_train =0, n_test=0;
            for(int t=0;t<prob.n_cv;t++){
                if(prob.idx_cv[t][i-1]!=j)
                    n_train++;
                else n_test++;
            }
            if(n_train+n_test!=prob.n || prob.n!= prob.n_cv){
                cout<<"n_train+n_test!= prob.n or prob.n!= prob.n_cv"<<endl;
                exit(1);
            }

            // get id_train, id_test, y_train, y_test
            int* id_train = Malloc(int,n_train);
            int* id_test  = Malloc(int,n_test);
            double* y_train = Malloc(double,n_train);
            double* y_test  = Malloc(double,n_test);
            for(int t=0,cnt_tr=0,cnt_te=0;t<prob.n_cv;t++){
                if(prob.idx_cv[t][i-1]!=j){
                    id_train[cnt_tr]  = t;
                    y_train[cnt_tr++] = prob.y[t];
                }else{
                    id_test[cnt_te]  = t;
                    y_test[cnt_te++] = prob.y[t];
                }
            }

            // search the best parameters by cross validation
            int tail = head+n_train-1;
            int* idx_j;
            get_column(idx_j, prob.idx_Asso, head,tail,prob.n_Asso,i-1);
            head = 1+tail;

            thread_pool[k-1].reset(new boost::thread(run_rs_fifo,id_train, n_train, id_test, n_test,k));
        }
    }

    //wait
    for(int i = 0; i < thread_pool.size(); i++)
    {
        thread_pool[i]->join();
    }

    rs_result.save_result(koil.rs_result_file,"KOIL_RS++");
    fifo_result.save_result(koil.fifo_result_file,"KOIL_FIFO++");
}

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cout<<"Argument format : "<<argv[0]<<"<dataset_file> <C> <gamma> <loss type>"<<endl;
        return 0;
    }

    KOIL koil;
    // load data
    koil.dataset_file = string(argv[1]);
    C = atof(argv[2]);
    g = atof(argv[3]);
    losstype = string(argv[4]);

    koil.load_data_path = "dataset/";

    koil.idx_asso_file = koil.dataset_file+"_IdxAsso";
    koil.idx_cv_file = koil.dataset_file+"_IdxCv";

    svm_problem& prob = koil.prob;
    prob.load_problem(koil.load_data_path+koil.dataset_file);
    prob.load_cross_validation(koil.load_data_path+koil.idx_asso_file,koil.load_data_path+koil.idx_cv_file);

    // save result path and file name
    koil.save_result_path="result/";
    koil.rs_model_file = koil.save_result_path+koil.dataset_file+"_rs_model.txt";
    koil.fifo_model_file = koil.save_result_path+koil.dataset_file+"_fifo_model.txt";

    koil.rs_result_file = koil.save_result_path+koil.dataset_file+"_rs_result.txt";
    koil.fifo_result_file = koil.save_result_path+koil.dataset_file+"_fifo_result.txt";

    koil.log_file = koil.save_result_path+koil.dataset_file+"_log.txt";

    online(koil);

    return 0;
}
