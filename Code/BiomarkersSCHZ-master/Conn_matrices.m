function [f] = Conn_matrices(dataset, group_1, group_2)
testfiledir = "C:\Users\Ilaria T\Desktop\Sapienza\Tesi Brain Network\Code\Dataset\" + dataset + "\" + group_1;
matfiles = dir(fullfile(testfiledir, '*.csv'));
nfiles = length(matfiles);

asd = [];
for i = 1:nfiles
   asd(:,:,i) = readmatrix(strcat(testfiledir, '\', matfiles(i).name));
end

for i = 1:nfiles
   asd(:,:,i) = triu(asd(:,:,i));
end

m = [];
for n = 1:nfiles
    A = asd(:,:,n);
    v = [];
    for i = 1:116
        for j = 1:116
            if j >= i 
                v = [v A(i,j)];
            end
        end
    end
    m(:,:, n) = v;
end
    
f_a = [];
for i = 1:nfiles
    f_a(:,i) = m(:,:,i)';
end

testfiledir = "C:\Users\Ilaria T\Desktop\Sapienza\Tesi Brain Network\Code\Dataset\" + dataset + "\" + group_2;
matfiles = dir(fullfile(testfiledir, '*.csv'));
nfiles = length(matfiles);

td = [];
for i = 1 : nfiles
   td(:,:,i) = readmatrix(strcat(testfiledir, '\', matfiles(i).name));
end

for i = 1:nfiles
   td(:,:,i) = triu(td(:,:,i));
end

m = [];
for n = 1:nfiles
    A = td(:,:,n);
    v = [];
    for i = 1:116
        for j = 1:116
            if j >= i 
                v = [v A(i,j)];
            end
        end
    end
    m(:,:, n) = v;
end
    
f_t = [];
for i = 1:nfiles
    f_t(:,i) = m(:,:,i)';
end

f = [f_a f_t];
save('features.mat', 'f');
end 
