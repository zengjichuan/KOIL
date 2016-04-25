#include "KOIL.h"
#include "utility.h"
#include "svm.h"
#include <ctime>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>
#include <random>
#include <chrono>
#define debug 0
#define debug_cv 0
#define log 0

using namespace std;

/****************************************
 * KOIL: Main Functions
 * **************************************/
// KOIL_RS++
void KOIL::rs_plus(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                   svm_model& model, double& AUC,double& Accuracy, double &Precision, double &Recall, double& time, int& err_count)
{
    // calculate AUC and accuracy on test set
    svm_node** x_test = Malloc(svm_node*,cnt_test);
    double* y_test = Malloc(double,cnt_test);
    double* AUC_test = Malloc(double,cnt_train);
    double* Acc_test = Malloc(double,cnt_train);
    for(int t=0;t<cnt_test;t++){
        x_test[t] = prob.x[id_test[t]];
        y_test[t] = prob.y[id_test[t]];
    }

    // initial prarameters
    int p,n,ridx;
    double at,ploss;
    clock_t start, end;
    bool flag;

    start = clock();
    err_count = 0;
    p=n=0;
    for(int t=0;t<cnt_train;t++){
        char buffer[100];
#if log
        sprintf(buffer, "The %d-th trial, Y(id)=%d",t+1,prob.y[id_train[t]]);
        write_log(buffer,0,string(buffer),string("result/Clog.txt"));
        //cout<<t<<"-th trial"<<endl;
#endif
        int num_block = cnt_train / 100;
        if (t%num_block == 0){
            cout << t / num_block << "%"<< endl;
        }
        // get the xt and yt
        svm_node* xt  = prob.x[id_train[t]];
        double& yt    = prob.y[id_train[t]];

        // predict xt
        double ft=model.predict(xt);
        if(ft*yt<=0)
            err_count++;

        if(yt==1)
        {
            p++;
            if(losstype=="l1")
                update_kernel_l1(xt,yt,model,at,ploss,true);
            else update_kernel_l2(xt,yt,model,at,ploss,true);
            rs_update_budget(xt,at,model.max_pos_n,p,
                             model,model.pos_alpha,model.pos_SV,model.pos_n,flag,ridx);
        }else{
            n++;
            if(losstype=="l1")
                update_kernel_l1(xt,yt,model,at,ploss,true);
            else update_kernel_l2(xt,yt,model,at,ploss,true);
            rs_update_budget(xt,at,model.max_neg_n,n,
                             model,model.neg_alpha,model.neg_SV,model.neg_n,flag,ridx);
        }

#if debug
        cout<<"rs pos_alpha:"<<endl;
        for(int i=0;i<model.pos_n;i++)
            cout<<model.pos_alpha[i]<<" ";
        cout<<endl;

        cout<<"rs neg_alpha:"<<endl;
        for(int i=0;i<model.neg_n;i++)
            cout<<model.neg_alpha[i]<<" ";
        cout<<endl<<endl;
#endif
        //cout<<"After update:flag = "<<flag<<", ridx="<<ridx<<", at = "<<at<<",pos_n="<<rs_model.pos_n<<endl<<endl;
        update_b(model);

#if debug
        cout<<"b = "<<model.b<<endl;
        cout<<endl;
        double* f_test = model.predict_list(x_test,cnt_test);
        evaluate_AUC(f_test,y_test,cnt_test,AUC_test[t],Acc_test[t]);
        cout<<t<<"-th trial:AUC="<<AUC_test[t]<<", Acc="<<Acc_test[t];
        cout<<endl;
#endif
#if log
        write_log(model.pos_alpha,model.pos_n,"Positive Ap","result/Clog.txt");
        write_log(model.neg_alpha,model.neg_n,"Negative Ap","result/Clog.txt");
        sprintf(buffer, "B: %.6f",model.b);
        write_log(buffer,0,string(buffer),string("result/Clog.txt"));
#endif
    }
    end = clock();
    time = (end-start)*1.0/CLOCKS_PER_SEC;

#if log
    write_log(AUC_test,cnt_train,"",this->log_file);
    write_log(Acc_test,cnt_train,"","result/acc.txt");
#endif

    // calculate AUC and accuracy on test set
    double* f_test = model.predict_list(x_test,cnt_test);
    evaluate_AUC(f_test,y_test,cnt_test,AUC,Accuracy, Precision, Recall);

    //free some arrays for memory
    free(f_test);
    free(y_test);
    free(x_test);
}

//KOIL_FIFO++
void KOIL::fifo_plus(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                   svm_model& model,double& AUC,double& Accuracy, double &Precision, double &Recall, double& time, int& err_count)
{
    // calculate AUC and accuracy on test set
    svm_node** x_test = Malloc(svm_node*,cnt_test);
    double* y_test = Malloc(double,cnt_test);
    for(int t=0;t<cnt_test;t++){
        x_test[t] = prob.x[id_test[t]];
        y_test[t] = prob.y[id_test[t]];
    }

    int p,n,ridx;
    double at,ploss;
    clock_t start, end;
    bool flag;

    start = clock();
    err_count = 0;
    p=n=0;
    for(int t=0;t<cnt_train;t++){
        // get the xt and yt
        int num_block = cnt_train / 100;
        if (t%num_block == 0){
            cout << t / num_block << "%"<< endl;
        }
        svm_node* xt  = prob.x[id_train[t]];
        double& yt    = prob.y[id_train[t]];

        // predict xt
        double ft=fifo_model.predict(xt);
        if(ft*yt<=0)
            err_count++;

        if(yt==1)
        {
            p++;
            if(losstype=="l1")
                update_kernel_l1(xt,yt,model,at,ploss,true);
            else update_kernel_l2(xt,yt,model,at,ploss,true);
            fifo_update_budget(xt,at,model.max_pos_n,model,model.fpidx,
                               model.pos_alpha,model.pos_SV,model.pos_n,flag,ridx);
        }else{
            n++;
            if(losstype=="l1")
                update_kernel_l1(xt,yt,model,at,ploss,true);
            else update_kernel_l2(xt,yt,model,at,ploss,true);
            fifo_update_budget(xt,at,model.max_neg_n,model,model.fnidx,
                               model.neg_alpha,model.neg_SV,model.neg_n,flag,ridx);
        }
        update_b(model);
    }
    end = clock();
    time = (end-start)*1.0/CLOCKS_PER_SEC;

    double* f_test = model.predict_list(x_test,cnt_test);
    evaluate_AUC(f_test,y_test,cnt_test,AUC,Accuracy,Precision, Recall);

    //free some arrays for memory
    free(f_test);
    free(y_test);
    free(x_test);
}

//KOIL_RS++ with Multiple Kernels - version 1: Each time update just one kernel
void KOIL::mkl_rs_plus(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                   svm_mkl& mkl,double& AUC,double& Accuracy, double &Precision, double &Recall, double& time, int& err_count)
{
    // calculate AUC and accuracy on test set
    svm_node** x_test = Malloc(svm_node*,cnt_test);
    double* y_test = Malloc(double,cnt_test);
    for(int t=0;t<cnt_test;t++){
        x_test[t] = prob.x[id_test[t]];
        y_test[t] = prob.y[id_test[t]];
    }

    int ridx;
    double at, ploss;
    clock_t start, end;
    bool flag;

    int test_cnt = 0;
    start = clock();
    err_count = 0;
    for(int i = 0; i < mkl.classifiers.size(); i++){
        mkl.classifiers[i].rsp = mkl.classifiers[i].rsn = 0;
    }

    // begin online training
    for(int t=0;t<cnt_train;t++){
        // get the xt and yt
        int num_block = cnt_train / 100;
        if (t%num_block == 0){
            cout << t / num_block << "%"<< endl;
        }
        svm_node* xt  = prob.x[id_train[t]];
        double& yt    = prob.y[id_train[t]];

        // predict xt
        double ft=mkl.predict(xt);
        if(ft*yt<=0)
            err_count++;

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        std::discrete_distribution<int> distribution(mkl.p.begin(),mkl.p.end());
        int kidx = distribution(generator);
        svm_model& model = mkl.classifiers[kidx];
        model.param.eta = mkl.lambda / mkl.p[kidx];

        if(yt==1)
        {
            model.rsp++;
            if(losstype=="l1")
                update_kernel_l1(xt,yt,model,at,ploss,false);
            else update_kernel_l2(xt,yt,model,at,ploss,false);
            rs_update_budget(xt,at,model.max_pos_n,model.rsp,
                             model,model.pos_alpha,model.pos_SV,model.pos_n,flag,ridx);
        }else{
            model.rsn++;
            if(losstype=="l1")
                update_kernel_l1(xt,yt,model,at,ploss,false);
            else update_kernel_l2(xt,yt,model,at,ploss,false);
            rs_update_budget(xt,at,model.max_neg_n,model.rsn,
                             model,model.neg_alpha,model.neg_SV,model.neg_n,flag,ridx);
        }
        update_b(model);
        mkl.weight[kidx] *= std::exp(-mkl.eta*(ploss)/mkl.p[kidx]);
        mkl.smooth_propbability();
    }
    end = clock();
    time = (end-start)*1.0/CLOCKS_PER_SEC;

    cout<<test_cnt<<endl;
    double* f_test = mkl.predict_list(x_test,cnt_test);
    evaluate_AUC(f_test,y_test,cnt_test,AUC,Accuracy,Precision,Recall);

    //free some arrays for memory
    free(f_test);
    free(y_test);
    free(x_test);
}

//KOIL_FIFO++ with Multiple Kernels - version 1: Each time update just one kernel
void KOIL::mkl_fifo_plus(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                   svm_mkl& mkl,double& AUC,double& Accuracy, double &Precision, double &Recall, double& time, int& err_count)
{
    std::ofstream fout;
    svm_node** x_test = Malloc(svm_node*,cnt_test);
    double* y_test = Malloc(double,cnt_test);
    for(int t=0;t<cnt_test;t++){
        x_test[t] = prob.x[id_test[t]];
        y_test[t] = prob.y[id_test[t]];
    }

    int ridx;
    double at, ploss;
    clock_t start, end;
    bool flag;

    start = clock();
    err_count = 0;

    //std::default_random_engine generator;
    for(int t=0;t<cnt_train;t++){
        // get the xt and yt
        int num_block = cnt_train / 100;
        if (t%num_block == 0){
            cout << t / num_block << "%"<< endl;
        }
        svm_node* xt  = prob.x[id_train[t]];
        double& yt    = prob.y[id_train[t]];

        // predict xt
        double ft=mkl.predict(xt);
        if(ft*yt<=0)
            err_count++;

        // construct a trivial random generator engine from a time-based seed:
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        std::discrete_distribution<int> distribution(mkl.p.begin(),mkl.p.end());

        int kidx = distribution(generator);
        svm_model& model = mkl.classifiers[kidx];
        model.param.eta = mkl.lambda / mkl.p[kidx];

        if(yt==1)
        {
            if(losstype=="l1")
                update_kernel_l1(xt,yt,model,at,ploss,false);
            else update_kernel_l2(xt,yt,model,at,ploss,false);
            fifo_update_budget(xt,at,model.max_pos_n,model,
                             model.fpidx,model.pos_alpha,model.pos_SV,model.pos_n,flag,ridx);
        }else{
            if(losstype=="l1")
                update_kernel_l1(xt,yt,model,at,ploss,false);
            else update_kernel_l2(xt,yt,model,at,ploss,false);
            fifo_update_budget(xt,at,model.max_neg_n,model,
                             model.fnidx,model.neg_alpha,model.neg_SV,model.neg_n,flag,ridx);
        }
        update_b(model);
        //double testtemp = model.f_norm();
        //double test1 = std::exp(-mkl.eta*(ploss+model.f_norm())/mkl.p[kidx]);
        mkl.weight[kidx] *= std::exp(-mkl.eta*(ploss)/mkl.p[kidx]);
//        for(int i = 0; i < mkl.p.size(); i++){
//            if(isnan(mkl.p[i])){
//                cout<<mkl.p[i]<<endl;
//            }
//        }
//        for(int i = 0; i < mkl.weight.size(); i++){
//            if(mkl.weight[i]==0){
//                cout<<ploss<<endl;
//            }
//            if(isnan(mkl.weight[i])){
//                cout<<mkl.weight[i]<<endl;
//                cout<<ploss<<endl;
//                cout<<mkl.p[i]<<endl;
//            }
//        }
        //mkl.normalize_weight();
        mkl.smooth_propbability();
        fout.open("./result/"+this->dataset_file+"_weight_mkl.txt",ios::app);
        for(int i = 0; i<mkl.weight.size(); i++){
            fout<<mkl.weight[i]<<"\t";
        }
        fout<<endl;
        fout.close();

    }
    end = clock();
    time = (end-start)*1.0/CLOCKS_PER_SEC;

    double* f_test = mkl.predict_list(x_test,cnt_test);
    evaluate_AUC(f_test,y_test,cnt_test,AUC,Accuracy,Precision, Recall);

    //free some arrays for memory
    free(f_test);
    free(y_test);
    free(x_test);
}

//KOIL_RS++ with Multiple Kernels - version 2: Update M classifiers at each trial with probability
void KOIL::mkl_rs_plus_m2(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                   svm_mkl& mkl,double& AUC,double& Accuracy, double &Precision, double &Recall, double& time, int& err_count)
{
    // calculate AUC and accuracy on test set
    svm_node** x_test = Malloc(svm_node*,cnt_test);
    double* y_test = Malloc(double,cnt_test);
    for(int t=0;t<cnt_test;t++){
        x_test[t] = prob.x[id_test[t]];
        y_test[t] = prob.y[id_test[t]];
    }

    int ridx;
    double at, max_weight,ploss;
    clock_t start, end;
    bool flag;

    int test_cnt = 0;
    start = clock();
    err_count = 0;
    for(int i = 0; i < mkl.classifiers.size(); i++){
        mkl.classifiers[i].rsp = mkl.classifiers[i].rsn = 0;
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    // begin online training
    for(int t=0;t<cnt_train;t++){
        // get the xt and yt
        int num_block = cnt_train / 100;
        if (t%num_block == 0){
            cout << t / num_block << "%"<< endl;
        }
        svm_node* xt  = prob.x[id_train[t]];
        double& yt    = prob.y[id_train[t]];

        // predict xt
        double ft=mkl.predict(xt);
        if(ft*yt<=0)
            err_count++;


        max_weight = *std::max_element(mkl.weight.begin(),mkl.weight.end());
        for(int i = 0; i < mkl.classifiers.size(); i++){
            std::bernoulli_distribution distribution(mkl.weight[i]/max_weight);
            if (distribution(generator)){
                test_cnt ++;
                if(yt==1){
                   mkl.classifiers[i].rsp++;
                   //update kernel
                   if(losstype=="l1"){
                       update_kernel_l1(xt,yt,mkl.classifiers[i],at,ploss,true);
                   }else update_kernel_l2(xt,yt,mkl.classifiers[i],at,ploss,true);

                   //update budget
                   rs_update_budget(xt,at,mkl.classifiers[i].max_pos_n,mkl.classifiers[i].rsp,
                                    mkl.classifiers[i],mkl.classifiers[i].pos_alpha,
                                    mkl.classifiers[i].pos_SV,mkl.classifiers[i].pos_n,flag,ridx);
                }else{
                    mkl.classifiers[i].rsn++;
                    //update kernel
                    if(losstype=="l1"){
                        update_kernel_l1(xt,yt,mkl.classifiers[i],at,ploss,true);
                    }else update_kernel_l2(xt,yt,mkl.classifiers[i],at,ploss,true);

                    //update budget
                    rs_update_budget(xt,at,mkl.classifiers[i].max_neg_n,mkl.classifiers[i].rsn,
                                       mkl.classifiers[i],mkl.classifiers[i].neg_alpha,
                                       mkl.classifiers[i].neg_SV,mkl.classifiers[i].neg_n,flag,ridx);
                }
                update_b(mkl.classifiers[i]);
                mkl.weight[i] *= std::exp(-mkl.eta*ploss);
            }
        }
        mkl.smooth_propbability();
    }
    end = clock();
    time = (end-start)*1.0/CLOCKS_PER_SEC;

    cout<<test_cnt<<endl;
    double* f_test = mkl.predict_list(x_test,cnt_test);
    evaluate_AUC(f_test,y_test,cnt_test,AUC,Accuracy,Precision,Recall);

    //free some arrays for memory
    free(f_test);
    free(y_test);
    free(x_test);
}

//KOIL_FIFO++ with Multiple Kernels - version 2: Update M classifiers at each trial with probability
void KOIL::mkl_fifo_plus_m2(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                   svm_mkl& mkl,double& AUC,double& Accuracy, double &Precision, double &Recall, double& time, int& err_count)
{
    std::ofstream fout;
    // calculate AUC and accuracy on test set
    svm_node** x_test = Malloc(svm_node*,cnt_test);
    double* y_test = Malloc(double,cnt_test);
    for(int t=0;t<cnt_test;t++){
        x_test[t] = prob.x[id_test[t]];
        y_test[t] = prob.y[id_test[t]];
    }

    int ridx;
    double at, max_weight, ploss;
    clock_t start, end;
    bool flag;

    start = clock();
    err_count = 0;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    for(int t=0;t<cnt_train;t++){
        // get the xt and yt
        int num_block = cnt_train / 100;
        if (t%num_block == 0){
            cout << t / num_block << "%"<< endl;
        }
        svm_node* xt  = prob.x[id_train[t]];
        double& yt    = prob.y[id_train[t]];

        // predict xt
        double ft=mkl.predict(xt);
        if(ft*yt<=0)
            err_count++;

        max_weight = *std::max_element(mkl.weight.begin(),mkl.weight.end());
        for(int i = 0; i < mkl.classifiers.size(); i++){
            std::bernoulli_distribution distribution(mkl.weight[i]/max_weight);
            if (distribution(generator)){
                if(yt==1){
                   //update kernel
                   if(losstype=="l1"){
                       update_kernel_l1(xt,yt,mkl.classifiers[i],at,ploss,true);
                   }else update_kernel_l2(xt,yt,mkl.classifiers[i],at,ploss,true);

                   //update budget
                   fifo_update_budget(xt,at,mkl.classifiers[i].max_pos_n,mkl.classifiers[i],
                                      mkl.classifiers[i].fpidx,mkl.classifiers[i].pos_alpha,
                                      mkl.classifiers[i].pos_SV,mkl.classifiers[i].pos_n,flag,ridx);
                }else{
                    //update kernel
                    if(losstype=="l1"){
                        update_kernel_l1(xt,yt,mkl.classifiers[i],at,ploss,true);
                    }else update_kernel_l2(xt,yt,mkl.classifiers[i],at,ploss,true);

                    //update budget
                    fifo_update_budget(xt,at,mkl.classifiers[i].max_neg_n,mkl.classifiers[i],
                                       mkl.classifiers[i].fnidx,mkl.classifiers[i].neg_alpha,
                                       mkl.classifiers[i].neg_SV,mkl.classifiers[i].neg_n,flag,ridx);
                }
                update_b(mkl.classifiers[i]);
                mkl.weight[i] *= std::exp(-mkl.eta*ploss);
            }
        }
        mkl.smooth_propbability();
        fout.open("./result/"+this->dataset_file+"_weight_mkl.txt",ios::app);
        for(int i = 0; i<mkl.weight.size(); i++){
            fout<<mkl.weight[i]<<"\t";
        }
        fout<<endl;
        fout.close();
    }
    end = clock();
    time = (end-start)*1.0/CLOCKS_PER_SEC;

    double* f_test = mkl.predict_list(x_test,cnt_test);
    evaluate_AUC(f_test,y_test,cnt_test,AUC,Accuracy,Precision,Recall);

    //free some arrays for memory
    free(f_test);
    free(y_test);
    free(x_test);
}

//KOIL_RS++ with Multiple Kernels - version 3: Update M classifiers at each trial with probability
void KOIL::mkl_rs_plus_m3(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                   svm_mkl& mkl,double& AUC,double& Accuracy, double &Precision, double &Recall, double& time, int& err_count)
{
    // calculate AUC and accuracy on test set
    svm_node** x_test = Malloc(svm_node*,cnt_test);
    double* y_test = Malloc(double,cnt_test);
    for(int t=0;t<cnt_test;t++){
        x_test[t] = prob.x[id_test[t]];
        y_test[t] = prob.y[id_test[t]];
    }

    int ridx;
    double at, max_weight,ploss;
    clock_t start, end;
    bool flag;

    int test_cnt = 0;
    start = clock();
    err_count = 0;
    for(int i = 0; i < mkl.classifiers.size(); i++){
        mkl.classifiers[i].rsp = mkl.classifiers[i].rsn = 0;
    }
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    // begin online training
    for(int t=0;t<cnt_train;t++){
        // get the xt and yt
        int num_block = cnt_train / 100;
        if (t%num_block == 0){
            cout << t / num_block << "%"<< endl;
        }
        svm_node* xt  = prob.x[id_train[t]];
        double& yt    = prob.y[id_train[t]];

        // predict xt
        double ft=mkl.predict(xt);
        if(ft*yt<=0)
            err_count++;


        //max_weight = *std::max_element(mkl.weight.begin(),mkl.weight.end());
        for(int i = 0; i < mkl.classifiers.size(); i++){
            std::bernoulli_distribution distribution(mkl.p[i]);
            if (distribution(generator)){
                test_cnt ++;
                if(yt==1){
                   mkl.classifiers[i].rsp++;
                   //update kernel
                   if(losstype=="l1"){
                       update_kernel_l1(xt,yt,mkl.classifiers[i],at,ploss,true);
                   }else update_kernel_l2(xt,yt,mkl.classifiers[i],at,ploss,true);

                   //update budget
                   rs_update_budget(xt,at,mkl.classifiers[i].max_pos_n,mkl.classifiers[i].rsp,
                                    mkl.classifiers[i],mkl.classifiers[i].pos_alpha,
                                    mkl.classifiers[i].pos_SV,mkl.classifiers[i].pos_n,flag,ridx);
                }else{
                    mkl.classifiers[i].rsn++;
                    //update kernel
                    if(losstype=="l1"){
                        update_kernel_l1(xt,yt,mkl.classifiers[i],at,ploss,true);
                    }else update_kernel_l2(xt,yt,mkl.classifiers[i],at,ploss,true);

                    //update budget
                    rs_update_budget(xt,at,mkl.classifiers[i].max_neg_n,mkl.classifiers[i].rsn,
                                       mkl.classifiers[i],mkl.classifiers[i].neg_alpha,
                                       mkl.classifiers[i].neg_SV,mkl.classifiers[i].neg_n,flag,ridx);
                }
                update_b(mkl.classifiers[i]);
                mkl.weight[i] *= std::exp(-mkl.eta*ploss);
            }
        }
        mkl.smooth_propbability();
    }
    end = clock();
    time = (end-start)*1.0/CLOCKS_PER_SEC;

    cout<<test_cnt<<endl;
    double* f_test = mkl.predict_list(x_test,cnt_test);
    evaluate_AUC(f_test,y_test,cnt_test,AUC,Accuracy,Precision,Recall);

    //free some arrays for memory
    free(f_test);
    free(y_test);
    free(x_test);
}

//KOIL_FIFO++ with Multiple Kernels - version 3: Update M classifiers at each trial with probability
void KOIL::mkl_fifo_plus_m3(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                   svm_mkl& mkl,double& AUC,double& Accuracy, double &Precision, double &Recall, double& time, int& err_count)
{
    std::ofstream fout;
    // calculate AUC and accuracy on test set
    svm_node** x_test = Malloc(svm_node*,cnt_test);
    double* y_test = Malloc(double,cnt_test);
    for(int t=0;t<cnt_test;t++){
        x_test[t] = prob.x[id_test[t]];
        y_test[t] = prob.y[id_test[t]];
    }

    int ridx;
    double at, max_weight, ploss;
    clock_t start, end;
    bool flag;

    start = clock();
    err_count = 0;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    for(int t=0;t<cnt_train;t++){
        // get the xt and yt
        int num_block = cnt_train / 100;
        if (t%num_block == 0){
            cout << t / num_block << "%"<< endl;
        }
        svm_node* xt  = prob.x[id_train[t]];
        double& yt    = prob.y[id_train[t]];

        // predict xt
        double ft=mkl.predict(xt);
        if(ft*yt<=0)
            err_count++;

        //max_weight = *std::max_element(mkl.weight.begin(),mkl.weight.end());
        for(int i = 0; i < mkl.classifiers.size(); i++){
            std::bernoulli_distribution distribution(mkl.p[i]);
            if (distribution(generator)){
                if(yt==1){
                   //update kernel
                   if(losstype=="l1"){
                       update_kernel_l1(xt,yt,mkl.classifiers[i],at,ploss,true);
                   }else update_kernel_l2(xt,yt,mkl.classifiers[i],at,ploss,true);

                   //update budget
                   fifo_update_budget(xt,at,mkl.classifiers[i].max_pos_n,mkl.classifiers[i],
                                      mkl.classifiers[i].fpidx,mkl.classifiers[i].pos_alpha,
                                      mkl.classifiers[i].pos_SV,mkl.classifiers[i].pos_n,flag,ridx);
                }else{
                    //update kernel
                    if(losstype=="l1"){
                        update_kernel_l1(xt,yt,mkl.classifiers[i],at,ploss,true);
                    }else update_kernel_l2(xt,yt,mkl.classifiers[i],at,ploss,true);

                    //update budget
                    fifo_update_budget(xt,at,mkl.classifiers[i].max_neg_n,mkl.classifiers[i],
                                       mkl.classifiers[i].fnidx,mkl.classifiers[i].neg_alpha,
                                       mkl.classifiers[i].neg_SV,mkl.classifiers[i].neg_n,flag,ridx);
                }
                update_b(mkl.classifiers[i]);
                mkl.weight[i] *= std::exp(-mkl.eta*ploss);
            }
        }
        mkl.smooth_propbability();
//        fout.open("./result/"+this->dataset_file+"_weight_mkl.txt",ios::app);
//        for(int i = 0; i<mkl.weight.size(); i++){
//            fout<<mkl.weight[i]<<"\t";
//        }
//        fout<<endl;
//        fout.close();
    }
    end = clock();
    time = (end-start)*1.0/CLOCKS_PER_SEC;

    double* f_test = mkl.predict_list(x_test,cnt_test);
    evaluate_AUC(f_test,y_test,cnt_test,AUC,Accuracy,Precision,Recall);

    //free some arrays for memory
    free(f_test);
    free(y_test);
    free(x_test);
}

//KOIL_RS++ with Multiple Kernels - Update M classifiers at each trial with probability
void KOIL::mkl_rs_plus_m(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                   svm_mkl& mkl,double& AUC,double& Accuracy, double &Precision, double &Recall, double& time, int& err_count)
{
    // calculate AUC and accuracy on test set
    svm_node** x_test = Malloc(svm_node*,cnt_test);
    double* y_test = Malloc(double,cnt_test);
    for(int t=0;t<cnt_test;t++){
        x_test[t] = prob.x[id_test[t]];
        y_test[t] = prob.y[id_test[t]];
    }

    int ridx;
    double at, max_weight,ploss;
    clock_t start, end;
    bool flag;

    int test_cnt = 0;
    start = clock();
    err_count = 0;
    for(int i = 0; i < mkl.classifiers.size(); i++){
        mkl.classifiers[i].rsp = mkl.classifiers[i].rsn = 0;
    }
    std::default_random_engine generator;
    // begin online training
    for(int t=0;t<cnt_train;t++){
        // get the xt and yt
        int num_block = cnt_train / 100;
        if (t%num_block == 0){
            cout << t / num_block << "%"<< endl;
        }
        svm_node* xt  = prob.x[id_train[t]];
        double& yt    = prob.y[id_train[t]];

        // predict xt
        double ft=mkl.predict(xt);
        if(ft*yt<=0)
            err_count++;


        max_weight = *std::max_element(mkl.weight.begin(),mkl.weight.end());
        for(int i = 0; i < mkl.classifiers.size(); i++){
            std::bernoulli_distribution distribution(mkl.weight[i]/max_weight);
            if (distribution(generator)){
                test_cnt ++;
                if(yt==1){
                   mkl.classifiers[i].rsp++;
                   //update kernel
                   if(losstype=="l1"){
                       update_kernel_l1(xt,yt,mkl.classifiers[i],at,ploss,true);
                   }else update_kernel_l2(xt,yt,mkl.classifiers[i],at,ploss,true);

                   //update budget
                   rs_update_budget(xt,at,mkl.classifiers[i].max_pos_n,mkl.classifiers[i].rsp,
                                    mkl.classifiers[i],mkl.classifiers[i].pos_alpha,
                                    mkl.classifiers[i].pos_SV,mkl.classifiers[i].pos_n,flag,ridx);
                }else{
                    mkl.classifiers[i].rsn++;
                    //update kernel
                    if(losstype=="l1"){
                        update_kernel_l1(xt,yt,mkl.classifiers[i],at,ploss,true);
                    }else update_kernel_l2(xt,yt,mkl.classifiers[i],at,ploss,true);

                    //update budget
                    rs_update_budget(xt,at,mkl.classifiers[i].max_neg_n,mkl.classifiers[i].rsn,
                                       mkl.classifiers[i],mkl.classifiers[i].neg_alpha,
                                       mkl.classifiers[i].neg_SV,mkl.classifiers[i].neg_n,flag,ridx);
                }
                update_b(mkl.classifiers[i]);
                if(at!=0){
                    mkl.weight[i] *=mkl.delta;
                }
            }
        }
        mkl.normalize_weight();
    }
    end = clock();
    time = (end-start)*1.0/CLOCKS_PER_SEC;

    cout<<test_cnt<<endl;
    double* f_test = mkl.predict_list(x_test,cnt_test);
    evaluate_AUC(f_test,y_test,cnt_test,AUC,Accuracy,Precision,Recall);

    //free some arrays for memory
    free(f_test);
    free(y_test);
    free(x_test);
}

//KOIL_FIFO++ with Multiple Kernels - Update M classifiers at each trial with probability
void KOIL::mkl_fifo_plus_m(int* id_train,int cnt_train, int* id_test, int cnt_test, string losstype,
                   svm_mkl& mkl,double& AUC,double& Accuracy, double &Precision, double &Recall, double& time, int& err_count)
{
    // calculate AUC and accuracy on test set
    svm_node** x_test = Malloc(svm_node*,cnt_test);
    double* y_test = Malloc(double,cnt_test);
    for(int t=0;t<cnt_test;t++){
        x_test[t] = prob.x[id_test[t]];
        y_test[t] = prob.y[id_test[t]];
    }

    int ridx;
    double at, max_weight, ploss;
    clock_t start, end;
    bool flag;

    start = clock();
    err_count = 0;

    std::default_random_engine generator;
    for(int t=0;t<cnt_train;t++){
        // get the xt and yt
        svm_node* xt  = prob.x[id_train[t]];
        double& yt    = prob.y[id_train[t]];

        // predict xt
        double ft=mkl.predict(xt);
        if(ft*yt<=0)
            err_count++;

        max_weight = *std::max_element(mkl.weight.begin(),mkl.weight.end());
        for(int i = 0; i < mkl.classifiers.size(); i++){
            std::bernoulli_distribution distribution(mkl.weight[i]/max_weight);
            if (distribution(generator)){
                if(yt==1){
                   //update kernel
                   if(losstype=="l1"){
                       update_kernel_l1(xt,yt,mkl.classifiers[i],at,ploss,true);
                   }else update_kernel_l2(xt,yt,mkl.classifiers[i],at,ploss,true);

                   //update budget
                   fifo_update_budget(xt,at,mkl.classifiers[i].max_pos_n,mkl.classifiers[i],
                                      mkl.classifiers[i].fpidx,mkl.classifiers[i].pos_alpha,
                                      mkl.classifiers[i].pos_SV,mkl.classifiers[i].pos_n,flag,ridx);
                }else{
                    //update kernel
                    if(losstype=="l1"){
                        update_kernel_l1(xt,yt,mkl.classifiers[i],at,ploss,true);
                    }else update_kernel_l2(xt,yt,mkl.classifiers[i],at,ploss,true);

                    //update budget
                    fifo_update_budget(xt,at,mkl.classifiers[i].max_neg_n,mkl.classifiers[i],
                                       mkl.classifiers[i].fnidx,mkl.classifiers[i].neg_alpha,
                                       mkl.classifiers[i].neg_SV,mkl.classifiers[i].neg_n,flag,ridx);
                }
                update_b(mkl.classifiers[i]);
                if(at!=0){
                    mkl.weight[i] *=mkl.delta;
                }
            }
        }
        mkl.normalize_weight();
    }
    end = clock();
    time = (end-start)*1.0/CLOCKS_PER_SEC;

    double* f_test = mkl.predict_list(x_test,cnt_test);
    evaluate_AUC(f_test,y_test,cnt_test,AUC,Accuracy,Precision,Recall);

    //free some arrays for memory
    free(f_test);
    free(y_test);
    free(x_test);
}

/****************************************
 * KOIL: Helper Functions
 * **************************************/
/**
 * @brief update the weight for SV
 *
 * @param xt the t-th sample xt
 * @param yt the label of xt
 * @param model the current decision function f
 * @return at return the weight of xt
 */
void KOIL::update_kernel_l1(svm_node* xt,double yt, svm_model& model, double& at, double& ploss, bool regularization)
{
    svm_node** same_sv;
    svm_node** oppo_sv;
    double* same_alpha;
    double* oppo_alpha;
    int same_n, oppo_n;
    ploss = 0;

    if(yt == 1){
       same_sv = model.pos_SV;
       same_alpha = model.pos_alpha;
       same_n = model.pos_n;
       oppo_sv = model.neg_SV;
       oppo_alpha = model.neg_alpha;
       oppo_n = model.neg_n;
    }else{
        oppo_sv = model.pos_SV;
        oppo_alpha = model.pos_alpha;
        oppo_n = model.pos_n;
        same_sv = model.neg_SV;
        same_alpha = model.neg_alpha;
        same_n = model.neg_n;
    }

    // make prediction to xt
    double ft = model.predict(xt);
    double* fb = Malloc(double,oppo_n);
    vector<double> loss;
    vector<pair<double,int>> simpair;
    memset(fb,0,oppo_n);
    // find the k-nearest opposite SV which violates the pairwise loss
    for(int i=0;i<oppo_n;i++){
        if(oppo_alpha[i]==0)
            continue;
        fb[i] = model.predict(oppo_sv[i]);
        double tloss = 1 - yt*(ft-fb[i]);
        if(tloss>0){
            simpair.push_back(pair<double,int> (model.kernel_func(xt,oppo_sv[i]),i));
        }
        loss.push_back(std::max(0.0,tloss));
    }

#if debug
    cout<<"Print fxt:"<<ft+model.b<<endl;
    cout<<"Print fb:"<<endl;
    for(int i=0;i<oppo_n;i++){
        // test: print fb[i]
        cout<<fb[i]+model.b<<" ";
    }
    cout<<endl;
#endif

    if(regularization){
        // (1-eta)*alpha degrade with eta
        for(int i=0;i<oppo_n;i++)
            oppo_alpha[i] *=(1-model.param.eta);
        for(int i=0;i<same_n;i++)
            same_alpha[i] *=(1-model.param.eta);
    }

    //Matlab
    if(oppo_n<1){
        at = model.param.C * model.param.eta * yt;
        ploss = 0;
        //ploss = std::min(std::max(0.0, 1.0-yt*ft),model.project_bound);
    }else{
        if(same_n<1 && simpair.size() == 0){
            at = model.param.C * model.param.eta * yt;
            ploss = 0;
            //ploss = std::min(std::max(0.0, 1.0-yt*ft),model.project_bound);
        }else{
            if(simpair.size()<model.k_num){
                at = simpair.size() * model.param.C * model.param.eta * yt;
                for(int i=0;i<simpair.size();i++){
                    oppo_alpha[simpair[i].second] -= model.param.C * model.param.eta * yt;
                    ploss += std::min(model.project_bound,loss[simpair[i].second]);
                }
            }else{
                at = model.k_num * model.param.C * model.param.eta * yt;
                sort(simpair.begin(),simpair.end(),comparator<double>);
                for(int i=0;i<model.k_num;i++){
                    oppo_alpha[simpair[i].second] -= model.param.C * model.param.eta * yt;
                    ploss += std::min(model.project_bound, loss[simpair[i].second]);
                }
            }
        }
    }

//    // another implementation
//    // update the weight of k-nearest opposite SV of xt
//    if(sim.size()>model.k_num){
//        std::vector<int> sidx;
//        ordered(sim,sidx);
//        for(int i=0;i<model.k_num;i++){
//            oppo_alpha[sidx[i]] -= model.param.C * model.param.eta * yt;
//        }
//    }
//    else{
//        for(int i=0;i<loss_idx.size();i++)
//            oppo_alpha[loss_idx[i]] -= model.param.C * model.param.eta * yt;
//    }

//    // calculate alpha_t for xt
//    if(oppo_n<1){
//        at = model.param.C * model.param.eta * yt;
//        //cout<<"enter, at="<<at<<endl;
//    }else{
//        if(same_n<1 && loss_idx.size() == 0){
//            at = model.param.C * model.param.eta * yt;
//            //cout<<"test, at="<<at<<endl;
//        }else
//            at = min((int)loss_idx.size(),model.k_num)*model.param.C * model.param.eta * yt;
//    }
    //cout<<"after update, at="<<at<<endl;
}

/**
 * @brief update the weight for SV based on smooth pairwise hinge loss
 *
 * @param xt the t-th sample xt
 * @param yt the label of xt
 * @param model the current decision function f
 * @return at return the weight of xt
 */
void KOIL::update_kernel_l2(svm_node* xt,double yt, svm_model& model, double& at, double& ploss, bool regularization)
{
    svm_node** same_sv;
    svm_node** oppo_sv;
    double* same_alpha;
    double* oppo_alpha;
    int same_n, oppo_n;
    ploss = 0;

    if(yt == 1){
       same_sv = model.pos_SV;
       same_alpha = model.pos_alpha;
       same_n = model.pos_n;
       oppo_sv = model.neg_SV;
       oppo_alpha = model.neg_alpha;
       oppo_n = model.neg_n;
    }else{
        oppo_sv = model.pos_SV;
        oppo_alpha = model.pos_alpha;
        oppo_n = model.pos_n;
        same_sv = model.neg_SV;
        same_alpha = model.neg_alpha;
        same_n = model.neg_n;
    }

    // make prediction to xt
    double ft = model.predict(xt);
    double* fb = Malloc(double,oppo_n);
    vector<double> loss;
    vector<pair<double,int>> simpair;
    memset(fb,0,oppo_n);
    // find the k-nearest opposite SV which violates the pairwise loss
    for(int i=0;i<oppo_n;i++){
        if(oppo_alpha[i]==0)
            continue;
        fb[i] = model.predict(oppo_sv[i]);
        double tloss = 1.0-yt*(ft-fb[i]);
        if(tloss>0){
            simpair.push_back(pair<double,int> (model.kernel_func(xt,oppo_sv[i]),i));
        }
        loss.push_back(std::max(0.0,tloss));
    }

    if(regularization){
        // (1-eta)*alpha degrade with eta
        for(int i=0;i<oppo_n;i++)
            oppo_alpha[i] *=(1-model.param.eta);
        for(int i=0;i<same_n;i++)
            same_alpha[i] *=(1-model.param.eta);
    }

    //Matlab
    if(oppo_n<1){
        at = model.param.C * model.param.eta * yt;
        ploss = 0;
//        ploss = std::min(std::max(0.0,1.0 - yt*ft),model.project_bound);
    }else{
        if(same_n<1 && simpair.size() == 0){
            at = model.param.C * model.param.eta * yt;
            ploss = 0;
//            ploss = std::min(std::max(0.0,1.0 - yt*ft),model.project_bound);
        }else{
            if(simpair.size()<model.k_num){
                at = 0;
                for(int i=0;i<simpair.size();i++){
                    //loss value
                    double temp = std::min(loss[simpair[i].second],model.project_bound);
                    at += 2*temp*model.param.C*model.param.eta*yt;
                    oppo_alpha[simpair[i].second] -= 2*temp*model.param.C * model.param.eta * yt;
                    ploss += (temp*temp);
                }
            }else{
                at = 0;
                sort(simpair.begin(),simpair.end(),comparator<double>);
#if debug
                for(int i=0;i<simpair.size();i++){
                    cout<<simpair[i].first<<",order :"<<simpair[i].second<<endl;
                }
#endif
                for(int i=0;i<model.k_num;i++){
                    //loss value
                    double temp = std::min(loss[simpair[i].second],model.project_bound);
                    at += 2*temp*model.param.C*model.param.eta*yt;
                    oppo_alpha[simpair[i].second] -= 2*temp*model.param.C * model.param.eta * yt;
                    ploss += (temp*temp);
                }
            }
        }
    }
}


/**
 * @brief KOIL_RS++: update budget
 *
 * @param xt xt the t-th sample xt
 * @param at the weight of xt
 * @param max_n the maximun number for the buffer
 * @param t the current iteration
 * @param model the current decision function f
 * @param alpha the weights of SVs, which have the same label with xt
 * @param SV the SV, which have the same label with xt
 * @param cur_n current number of SVs in the buffer
 * @param flag indicate whether xt is put in the buffer or not
 * @param ridx the replaced index for xt if xt is put in the buffer
 */
void KOIL::rs_update_budget(svm_node* xt,double at,int max_n,int t,
                            svm_model& model, double* &alpha,svm_node** &SV,int& cur_n,bool& flag, int& ridx)
{
    //    cout<<"at="<<at<<",t="<<t<<endl;
    // variable initialization
    ridx = -1;
    flag = false;
    double ac = 0;
    svm_node* svc = NULL;

    if(at==0)
        return;
    if(cur_n<max_n)
    {
        alpha[cur_n] = at;
        SV[cur_n] = xt;
        flag = true;
        cur_n ++;
    }else{
        // replace one instance from SV with Probability = max_n/t
        //cout<<"t="<<t<<endl;
        srand((unsigned)time(0));
        //int tempind = rand()%t;
        //cout<<tempind;
        if(rand()%t<max_n){
            ridx = rand()%max_n;
            ac = alpha[ridx];
            svc = SV[ridx];
            alpha[ridx] = at;
            SV[ridx]=xt;
            flag = true;
        }else{
            ac = at;
            svc = xt;
        }

        // find the most similar SV for compensation
        int cidx = 0;
        double ma = 0;
        for(int i=0;i<max_n;i++){
            double temp = model.kernel_func(svc,SV[i]);
            if(temp>ma){
                cidx = i;
                ma = temp;
            }
        }
        alpha[cidx] += ac;
    }
}

/**
 * @brief KOIL_FIFO++: update budget
 *
 * @param xt xt the t-th sample xt
 * @param at the weight of xt
 * @param max_n the maximun number for the buffer
 * @param model the current decision function f
 * @param fidx the index of the first SV in the buffer (FIFO)
 * @param alpha the weights of SVs, which have the same label with xt
 * @param SV the SV, which have the same label with xt
 * @param cur_n current number of SVs in the buffer
 * @param flag indicate whether xt is put in the buffer or not
 * @param ridx the replaced index for xt if xt is put in the buffer
 */
void KOIL::fifo_update_budget(svm_node* xt,double at,int max_n,
                              svm_model& model,int& fidx,double* &alpha,svm_node** &SV,int& cur_n,bool& flag, int& ridx)
{
    // variable initialization
    ridx = -1;
    flag = false;
    double ac = 0;
    svm_node* svc = NULL;

    if(at==0)
        return;
    if(cur_n<max_n)
    {
            alpha[cur_n] = at;
            SV[cur_n] = xt;
            flag = true;
            cur_n ++;
    }else{
        // replace the first SV in the buffer by fpidx and fnidx
        ac = alpha[fidx];
        svc = SV[fidx];
        alpha[fidx] = at;
        SV[fidx] = xt;
        flag = true;
        ridx = fidx;
        fidx = (fidx+1) % max_n;

        // find the most similar SV for compensation
        int cidx = 0;
        double ma = 0;
        for(int i=0;i<max_n;i++){
            double temp = model.kernel_func(svc,SV[i]);
            if(temp>ma){
                cidx = i;
                ma = temp;
            }
        }
//        cout<<"The maximun sim: "<<ma<<" , idx: "<<cidx<<endl;
        alpha[cidx] += ac;
    }
}
/*
void KOIL::fifo_update_budget(svm_node* xt,double at,int max_n,
                              svm_model& model,int& fidx,double* &alpha,svm_node** &SV,int& cur_n,bool& flag, int& ridx)
{
    // variable initialization
    int max_n, cur_n;
    double* alpha;
    svm_node** SV;

    ridx = -1;
    flag = false;
    double ac = 0;
    svm_node* svc = NULL;

    if(at==0)
        return;
    if(at>0){
        max_n = model.max_pos_n;
        fidx = model.fpidx;
        alpha = model.pos_alpha;
        SV = model.pos_SV;
        cur_n = model.pos_n;
    }else{
        max_n = model.max_neg_n;
        fidx = model.fnidx;
        alpha = model.neg_alpha;
        SV = model.neg_SV;
        cur_n = model.neg_n;
    }

    if(cur_n<max_n)
    {
            alpha[cur_n] = at;
            SV[cur_n] = xt;
            flag = true;
            //cur_n ++;
            if(at>0)
                model.pos_n++;
            else model.neg_n++;
    }else{
        // replace the first SV in the buffer by fpidx and fnidx
        ac = alpha[fidx];
        svc = SV[fidx];
        alpha[fidx] = at;
        SV[fidx] = xt;
        flag = true;
        ridx = fidx;
        fidx = (fidx+1) % max_n;

        // find the most similar SV for compensation
        int cidx = 0;
        double ma = 0;
        for(int i=0;i<max_n;i++){
            double temp = model.kernel_func(svc,SV[i]);
            if(temp>ma){
                cidx = i;
                ma = temp;
            }
        }
//        cout<<"The maximun sim: "<<ma<<" , idx: "<<cidx<<endl;
        alpha[cidx] += ac;
    }
}
*/

/**
 * @brief KOIL: update threshold of decision function
 *
 * @param model the current decision function
 */
void KOIL::update_b(svm_model& model)
{
    if(model.pos_n==0||model.neg_n==0){
        model.b=0;
        return;
    }
    model.l = model.pos_n+model.neg_n;
    double* f_pos = model.predict_list(model.pos_SV,model.pos_n);
    double* f_neg = model.predict_list(model.neg_SV,model.neg_n);

    // find the min of positive value, max of negative value
    double pmin=f_pos[0], nmax=f_neg[0];
    for(int i=0;i<model.pos_n;i++){
        if(pmin>f_pos[i])
            pmin = f_pos[i];
    }

    for(int i=0;i<model.neg_n;i++){
        if(nmax<f_neg[i])
            nmax = f_neg[i];
    }

    // find the threshold
    int berr;
    double current_b = model.b;
    if(pmin>=nmax){
        model.b = current_b + (pmin+nmax)/2;
        berr = 0;
    }else{
        double bstep = (nmax-pmin)/500;
        berr = model.pos_n+model.neg_n;
        for(double cb=pmin+bstep;cb<nmax;cb+=bstep){
            int err = 0;
            for(int i=0;i<model.neg_n;i++)
                if(f_neg[i]-cb>0)
                    err++;
            for(int i=0;i<model.pos_n;i++)
                if(f_pos[i]-cb<0)
                    err++;
            if(berr > err){
                model.b = current_b + cb;
                berr = err;
            }
        }
    }
//    cout<<"After update b"<<endl<<"positive predicted";
//    f_pos = model.predict_list(model.pos_SV,model.pos_n);
//    f_neg = model.predict_list(model.neg_SV,model.neg_n);
//    for(int i=0;i<model.pos_n;i++){
//        cout<<f_pos[i]<<" ";
//    }
//    cout<<endl<<"negative predicted"<<endl;
//    for(int i=0;i<model.neg_n;i++){
//        cout<<f_neg[i]<< " ";
//    }
//    cout<<endl;
//    cout<<"smallest error:"<<berr<<endl;
//    cout<<"model.b: "<<model.b<<endl;
}

/**
 * @brief the calculate the AUC and Accuracy between f and y
 *
 * @param f 1xn vector, the predicted label by the model
 * @param y 1xn vector, the true label
 * @param n the number of the label
 * @return AUC AUC value
 * @return Accuracy Accuracy for the correct prediction
 */
void KOIL::evaluate_AUC(double* f, double* y, int n,
                        double& AUC, double& Accuracy, double& Precision, double &Recall)
{
    int correct = 0;
    int num_pos = 0;
    int num_neg = 0;
    int c = 0, tp = 0, p = 0;
    for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            if((y[i]-y[j])*(f[i]-f[j])>0)
                correct ++;
        }
        if(y[i]>0)
            num_pos++;

        else num_neg++;
        if(f[i]*y[i]>=0)
            c++;
        if(y[i]>0 && f[i] > 0)
            tp ++;
        if(f[i]>0)
            p++;


    }
    AUC = correct*1.0/(num_neg*num_pos);
    Precision = tp *1.0/p;
    Recall = tp *1.0/num_pos;
    Accuracy = c*1.0/n;
}


/****************************************
 * koil_result: save and load KOIL result
 * **************************************/
// initial koil result
void koil_result::initial_result(int n)
{
    this->runs = n;
    this->auc  = Malloc(double,runs);
    this->accuracy = Malloc(double,runs);
    this->precision = Malloc(double, runs);
    this->recall = Malloc(double, runs);
    this->time = Malloc(double,runs);
    this->err_cnt = Malloc(int,runs);
}

// free the memory of the result
void koil_result::free_result()
{
    free(this->auc);
    free(this->accuracy);
    free(this->precision);
    free(this->recall);
    free(this->time);
    free(this->err_cnt);
}

template<class T>
// calculate the mean and std
void mean_std(T* x,int n, T& m, T& sig){
    m=0;
    for(int i=0;i<n;i++)
        m += x[i];
    m/=n;

    sig = 0;
    for(int i=0;i<n;i++){
        sig += (x[i]-m)*(x[i]-m);
    }
    sig = std::sqrt(sig/n);
}

// save koil result
void koil_result::save_result(string path, string method)
{
    cout<< "save_result path :"<<path<<endl;
    cout<<" method :"<<method<<endl;
    double mean_auc, mean_accuracy, mean_precision , mean_recall, mean_time, std_auc, std_accuracy, std_precision, std_recall, std_time;
    int mean_err_cnt, std_err_cnt;
    mean_std(this->auc,this->runs,mean_auc,std_auc);
    mean_std(this->accuracy,this->runs,mean_accuracy,std_accuracy);
    mean_std(this->precision,this->runs,mean_precision,std_precision);
    mean_std(this->recall,this->runs,mean_recall,std_recall);
    mean_std(this->time,this->runs,mean_time,std_time);
    mean_std(this->err_cnt,this->runs,mean_err_cnt,std_err_cnt);
    std::ofstream ofresult;
    ofresult.open(path,ios::app);
    for(int i=0;i<this->runs;i++)
        ofresult<<this->auc[i]<<"\t";
    ofresult<<endl;
    for(int i=0;i<this->runs;i++)
        ofresult<<this->accuracy[i]<<"\t";
    ofresult<<endl;
    for(int i=0;i<this->runs;i++)
        ofresult<<this->precision[i]<<"\t";
    ofresult<<endl;
    for(int i=0;i<this->runs;i++)
        ofresult<<this->recall[i]<<"\t";
    ofresult<<endl;
    for(int i=0;i<this->runs;i++)
        ofresult<<this->time[i]<<"\t";
    ofresult<<endl;
    for(int i=0;i<this->runs;i++)
        ofresult<<this->err_cnt[i]<<"\t";
    ofresult<<endl;

    ofresult<<"------------------------------------------------------------------------------------"<<endl;
    ofresult<<"method\t &\t AUC\t &Accuracy\t&Precision\t&Recall\t &Time\t &error counts \\\\ \n";
    ofresult<<method<<"\t &\t "<<mean_auc<<"$\\pm$"<<std_auc<<"\t &\t "
           <<mean_accuracy<<"$\\pm$"<<std_accuracy << "\t &\t "
          <<mean_precision<<"$\\pm$"<<std_precision << "\t &\t "
          <<mean_recall<<"$\\pm$"<<std_recall<<"\t &\t "
           <<mean_time<<"$\\pm$"<<std_time<<"\t &\t "
           <<mean_err_cnt<<"$\\pm$"<<std_err_cnt<<endl;
    ofresult<<"------------------------------------------------------------------------------------";
    ofresult.close();
//    fprintf(fp,"------------------------------------------------------------------------------------");
//    fprintf(fp,"method\t &\t AUC\t &Accuracy\t &Time\t &error counts \\\\ \n");
//    fprintf(fp,"%s\t &\t %.3f$\pm$%.3f \t &%.3f$\pm$%.3f\t &%.3f$\pm$%.3f\t &%.3f$\pm$%.3f \n",method.c_str(),mean_auc,std_auc,
//            mean_accuracy,std_accuracy,mean_time,std_time,mean_err_cnt,std_err_cnt);
//    fprintf(fp,"------------------------------------------------------------------------------------");
}

// load koil result (unfinished)
void koil_result::load_result(string path){}



