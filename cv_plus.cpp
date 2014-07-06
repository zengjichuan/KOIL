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

boost::mutex brs_auc_mutex;
boost::mutex best_c_mutex;
boost::mutex best_g_mutex;
boost::mutex file_mutex;

int cnum, gnum;
double cstart, gstart;
double cstep, gstep;
int cvfold;
string losstype;

void parallel_cross_validation(KOIL& koil, int*& id_x, double*& y, int*& idx_j, int n_data)
{
    svm_model rs_model;
    svm_model fifo_model;
    // the test list for C and gamma
    //int cnum = 10, gnum = 10;
    cout<<"cnum"<<cnum<<endl;
    cout<<"gnum"<<gnum<<endl;
//    cstep = 2;
//    gstep = 4;
    double* clist = Malloc(double,cnum);
    double* glist = Malloc(double,gnum);
    double** auc_record=Malloc(double*,cnum);
    double** acc_record=Malloc(double*,cnum);
    for(int i=0;i<cnum;i++){
        auc_record[i] = Malloc(double,gnum);
        acc_record[i] = Malloc(double,gnum);
    }

    clist[0] = cstart;
    glist[0] = gstart;
    for(int i=1;i<cnum;clist[i] = clist[i-1]+cstep,i++);
    for(int j=1;j<gnum;glist[j] = glist[j-1]+gstep,j++);

    double best_rs_C, best_fifo_C, best_rs_gamma, best_fifo_gamma;
    best_rs_C = best_fifo_C = best_rs_gamma = best_fifo_gamma = 0;
    double brs_auc = 0;
    double rs_time;
    int rs_err_cnt;

    auto run_function = [&](int c_index, int g_index)
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
            svm_model rs;
            rs.initialize(100);
            rs.param.C=clist[c_index];
            rs.param.gamma=glist[g_index];
            koil.rs_plus(id_train, n_train, id_test, n_test,losstype,
                    rs,
                    AUC_RS[fold-1],Acc_RS[fold-1],
                    rs_time,rs_err_cnt);

            //2. KOIL_FIFO++

        }
        auc_record[c_index][g_index]=0;
        acc_record[c_index][g_index]=0;
        for(int tcnt = 0; tcnt<cvfold;tcnt++ ){
            auc_record[c_index][g_index] += AUC_RS[tcnt];
            acc_record[c_index][g_index] += Acc_RS[tcnt];
        }
        auc_record[c_index][g_index] /= cvfold;
        acc_record[c_index][g_index] /= cvfold;
        cout<<"Avg AUC = "<<auc_record[c_index][g_index]<<endl;
        cout<<"Avg Acc = "<<acc_record[c_index][g_index]<<endl;

        brs_auc_mutex.lock();
        best_c_mutex.lock();
        best_g_mutex.lock();
        if(brs_auc<auc_record[c_index][g_index]){
            brs_auc = auc_record[c_index][g_index];
            best_rs_C = clist[c_index];
            best_rs_gamma = glist[g_index];
        }
        brs_auc_mutex.unlock();
        best_c_mutex.unlock();
        best_g_mutex.unlock();
    };

    std::vector<boost::shared_ptr<boost::thread>> thread_pool(cnum*gnum);
    // loop through clist and glist
    for(int i=0;i<cnum;i++)
    {
        cout<<"CV: C="<<clist[i]<<endl;
        for(int j=0;j<gnum;j++)
        {
            cout<<"CV: gamma="<<glist[j]<<endl;
            thread_pool[i * gnum + j].reset(new boost::thread(run_function, i, j));
        }
    }
    //wait
    for(int i = 0; i < thread_pool.size(); i++)
    {
        thread_pool[i]->join();
    }

    fifo_model.param.C = rs_model.param.C = best_rs_C;
    fifo_model.param.gamma = rs_model.param.gamma = best_rs_gamma;

    write_matrix(auc_record,cnum,gnum,"result/"+koil.dataset_file+"_cv_AUC.txt");
    write_matrix(acc_record,cnum,gnum,"result/"+koil.dataset_file+"_cv_Acc.txt");

    std::ofstream of("result/"+koil.dataset_file+"_cv.txt",ios::app);
    of<<"The best AUC = "<<brs_auc<<endl;
    of<<"The best C = "<<best_rs_C<<endl;
    of<<"The best gamma = "<<best_rs_gamma<<endl;
    of.close();

    std::ofstream ofs("result/"+koil.dataset_file+"_cv_c.txt",ios::app);
    for(int i=0;i<cnum;i++){
    ofs<<clist[i]<<"\t";
    }
    ofs<<endl<<endl;
    ofs.close();


    std::ofstream ofg("result/"+koil.dataset_file+"_cv_g.txt",ios::app);
    for(int i=0;i<cnum;i++){
    ofg<<glist[i]<<"\t";
    }
    ofg<<endl<<endl;
    ofg.close();

    std::ofstream ofa("result/"+koil.dataset_file+"_cv_pair.txt",ios::app);
    for(int i=0;i<cnum;i++){
    for(int j=0;j<gnum;j++){
        ofa<<clist[i]<<"\t"<<glist[j]<<"\t"<<auc_record[i][j]<<"\t"<<acc_record[i][j]<<endl;
    }
    }
    ofa<<endl<<endl;
    ofa.close();

}

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


int main(int argc, char **argv)
{
    if(argc != 10)
    {
        cout<<"Argument format : "<<argv[0]<<"<dataset_file>  <Num of clist> <Num of glist> <c start> <g start> <cstep><gstep> <cvfold> <loss type>"<<endl;
        return 0;
    }

    KOIL koil;
    // load data
    koil.dataset_file = string(argv[1]);
    cnum = atoi(argv[2]);
    gnum = atoi(argv[3]);
    cstart = atof(argv[4]);
    gstart = atof(argv[5]);
    cstep = atof(argv[6]);
    gstep = atof(argv[7]);
    cvfold = atoi(argv[8]);
    losstype = string(argv[9]);
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

