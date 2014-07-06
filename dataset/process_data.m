function process_data
dataset = {'glass';'vowel';'vehicle';'svmguide4';'svmguide3';'svmguide2';'svmguide1_sub';'segment';
          'satimage_sub'; 'liver_disorders';'ionosphere';'heart';'german';'fourclass';
          'dna';'diabetes';'breast_cancer';'australian';}
      
% ds1={ 'glass';'german';'vowel';'satimage_sub';'segment';'diabetes';'svmguide2';};
% ds2={ 'svmguide1_sub';}
% ds3={'ionosphere';'fourclass';'dna'; };
%           %'heart';'breast_cancer';'australian';'vehicle';'svmguide4';'svmguide3';'liver_disorders';
% dataset=ds2;          
      
n=size(dataset,1);

for i=1:1:n
    t=cellstr(dataset(i));
    st=char(t);
    load_path = './real_data/';
    save_path = './real_data/libsvm_data/';
    libsvm_write([load_path st],[save_path st]);

end

end