/**
 * @brief Kernelized Online Imbalanced Learning with Fixed Buddget
 * Implemented by Junjie Hu
 * Contact: jjhu@cse.cuhk.edu.hk
*/

#include "svm.h"
#include "KOIL.h"
#include "utility.h"
#include "cv.h"
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

boost::mutex brs_auc_mutex;
boost::mutex best_c_mutex;
boost::mutex best_eta_mutex;
boost::mutex file_mutex;

int cnum, etanum;
int cstart, etastart;
int cstep, etastep;
int cvfold;
string losstype;
double smooth_delta;

void parallel_cross_validation(KOIL& koil, int*& id_x, double*& y, int*& idx_j, int n_data)
{
    //svm_model rs_model;
    //svm_model fifo_model;
    // the test list for C and gamma
    //int cnum = 10, gnum = 10;
    cout<<"cnum"<<cnum<<endl;
    cout<<"etanum"<<etanum<<endl;
//    cstep = 2;
//    gstep = 4;
    double* clist = Malloc(double,cnum);
    double* etalist = Malloc(double,etanum);
    double** auc_record=Malloc(double*,cnum);
    double** acc_record=Malloc(double*,cnum);
    for(int i=0;i<cnum;i++){
        auc_record[i] = Malloc(double,etanum);
        acc_record[i] = Malloc(double,etanum);
    }

    clist[0] = pow(2,cstart);
    etalist[0] = pow(2,etastart);
    for(int i=1;i<cnum;clist[i] = clist[i-1]*cstep,i++);
    for(int j=1;j<etanum;etalist[j] = etalist[j-1]*etastep,j++);

    double best_rs_C, best_fifo_C, best_rs_eta, best_fifo_gamma;
    best_rs_C = best_fifo_C = best_rs_eta = best_fifo_gamma = 0;
    double brs_auc = 0;
    double rs_time;
    int rs_err_cnt;

    auto run_function = [&](int c_index, int eta_index)
    {
        double AUC_RS[]={0,0,0,0,0};
        double Acc_RS[]={0,0,0,0,0};
        // Five fold cross-validation
        for(int fold=1;fold<=cvfold;fold++){
            cout<<"Fold:"<<fold<<endl;
            int n_train =0, n_test=0;
            for(int t=0;t<n_data;t++){
                if(idx_j[t]!=fold)
                    n_train++;
                else n_test++;
            }
            if(n_train+n_test != n_data){
                cout<<"CrossValidation:n_train+n_test!= n_data"<<endl;
                exit(1);
            }
            int* id_train = Malloc(int,n_train);
            int* id_test  = Malloc(int,n_test);
            double* y_train = Malloc(double,n_train);
            double* y_test  = Malloc(double,n_test);

            // get id_train, id_test, y_train, y_test
            for(int t=0,cnt_tr=0,cnt_te=0;t<n_data;t++){
                if(idx_j[t]!=fold){
                    id_train[cnt_tr]  = id_x[t];
                    y_train[cnt_tr++] = y[t];
                }else{
                    id_test[cnt_te]  = id_x[t];
                    y_test[cnt_te++] = y[t];
                }
            }

            //1. KOIL_RS++
            svm_mkl mkl;
            mkl.eta = etalist[eta_index];
            vector<double> glist;
            vector<int> degreelist;
            for(int i=-6; i<=6;i++) glist.push_back(std::pow(2,i));
            for(int i=1; i<=3;i++) degreelist.push_back(i);
            mkl.initialize(100,smooth_delta,clist[c_index],glist,degreelist);
            koil.mkl_rs_plus_m2(id_train, n_train, id_test, n_test, losstype,
                             mkl,
                             AUC_RS[fold-1],Acc_RS[fold-1],
                             rs_time,rs_err_cnt);


            //2. KOIL_FIFO++

        }
        auc_record[c_index][eta_index]=0;
        acc_record[c_index][eta_index]=0;
        for(int tcnt = 0; tcnt<cvfold;tcnt++ ){
            auc_record[c_index][eta_index] += AUC_RS[tcnt];
            acc_record[c_index][eta_index] += Acc_RS[tcnt];
        }
        auc_record[c_index][eta_index] /= cvfold;
        acc_record[c_index][eta_index] /= cvfold;
        cout<<"Avg AUC = "<<auc_record[c_index][eta_index]<<endl;
        cout<<"Avg Acc = "<<acc_record[c_index][eta_index]<<endl;

        brs_auc_mutex.lock();
        best_c_mutex.lock();
        best_eta_mutex.lock();
        if(brs_auc<auc_record[c_index][eta_index]){
            brs_auc = auc_record[c_index][eta_index];
            best_rs_C = clist[c_index];
            best_rs_eta = etalist[eta_index];
        }
        brs_auc_mutex.unlock();
        best_c_mutex.unlock();
        best_eta_mutex.unlock();
    };

    std::vector<boost::shared_ptr<boost::thread>> thread_pool(cnum*etanum);
    // loop through clist and etalist
    for(int i=0;i<cnum;i++)
    {
        cout<<"CV: C="<<clist[i]<<endl;
        for(int j=0;j<etanum;j++)
        {
            cout<<"CV: gamma="<<etalist[j]<<endl;
            thread_pool[i * etanum + j].reset(new boost::thread(run_function, i, j));
        }
    }
    //wait
    for(int i = 0; i < thread_pool.size(); i++)
    {
        thread_pool[i]->join();
    }

    //fifo_model.param.C = rs_model.param.C = best_rs_C;
    //fifo_model.param.gamma = rs_model.param.gamma = best_rs_gamma;

    write_matrix(auc_record,cnum,etanum,"result/"+koil.dataset_file+"_cv_AUC.txt");
    write_matrix(acc_record,cnum,etanum,"result/"+koil.dataset_file+"_cv_Acc.txt");

    std::ofstream of("result/"+koil.dataset_file+"_cv.txt",ios::app);
    of<<"The best AUC = "<<brs_auc<<endl;
    of<<"The best C = "<<best_rs_C<<endl;
    of<<"The best eta = "<<best_rs_eta<<endl;
    of.close();

//    std::ofstream ofs("result/"+koil.dataset_file+"_cv_c.txt",ios::app);
//    for(int i=0;i<cnum;i++){
//    ofs<<clist[i]<<"\t";
//    }
//    ofs<<endl<<endl;
//    ofs.close();


//    std::ofstream ofg("result/"+koil.dataset_file+"_cv_g.txt",ios::app);
//    for(int i=0;i<cnum;i++){
//    ofg<<etalist[i]<<"\t";
//    }
//    ofg<<endl<<endl;
//    ofg.close();

    std::ofstream ofa("result/"+koil.dataset_file+"_cv_pair.txt",ios::app);
    for(int i=0;i<cnum;i++){
    for(int j=0;j<etanum;j++){
        ofa<<clist[i]<<"\t"<<etalist[j]<<"\t"<<auc_record[i][j]<<"\t"<<acc_record[i][j]<<endl;
    }
    }
    ofa<<endl<<endl;
    ofa.close();

}


int main(int argc, char **argv)
{
    if(argc != 11)
    {
        cout<<"Argument format : "<<argv[0]<<"<dataset_file> <Num of clist> <Num of etalist> <c start> <eta start> <cstep> <etastep> <cvfold> <loss type> <smooth delta> "<<endl;
        return 0;
    }

    KOIL koil;
    // load data
    koil.dataset_file = string(argv[1]);
    cnum = atoi(argv[2]);
    etanum = atoi(argv[3]);
    cstart = atoi(argv[4]);
    etastart = atoi(argv[5]);
    cstep = atoi(argv[6]);
    etastep = atoi(argv[7]);
    cvfold = atoi(argv[8]);
    losstype = string(argv[9]);
    smooth_delta = atof(argv[10]);
    cout<<"cvfold"<<cvfold<<endl;
    koil.load_data_path = "dataset/";
    //koil.dataset_file = "w8a";


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

    int* id_x, *idx_j, n_train;
    double* y;
    get_cv_list(koil,id_x,y,idx_j,n_train);
    parallel_cross_validation(koil,id_x,y,idx_j,n_train);
    return 0;
}

