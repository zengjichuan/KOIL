function libsvm_write(load_file,save_file)

load(strcat(load_file,'.mat'));

fid=fopen(save_file,'w');
[n d] = size(data);
for i=1:n
    fprintf(fid,'%d\t',data(i,1));
    for j=2:d
        if(data(i,j)~=0)
            fprintf(fid,'%d:%.8f\t',j-1,data(i,j));
        end
    end
    fprintf(fid,'\n');
end
fclose(fid);

fid = fopen(strcat(save_file,'_IdxCv'),'w');
[in id] = size(IdxStr.IdxCv);
for i =1:in
    for j = 1:id
        fprintf(fid,'%d\t',IdxStr.IdxCv(i,j));
    end
    fprintf(fid,'\n');
end
fclose(fid);

fid = fopen(strcat(save_file,'_IdxAsso'),'w');
[ian iad] = size(IdxStr.IdxAsso);
for i = 1:ian
    for j = 1:iad
        fprintf(fid,'%d\t',IdxStr.IdxAsso(i,j));
    end
    fprintf(fid,'\n');
end
fclose(fid);
end