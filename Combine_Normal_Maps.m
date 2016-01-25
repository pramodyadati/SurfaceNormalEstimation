%function [nm]=Combine_Normal_Maps(img,normals,num)
function [nm]=Combine_Normal_Maps(img,num)

load('data/cnnNormal-baseline/net-epoch-200.mat');
run ../matconvnet-master/matlab/vl_setupnn.m;
net.layers=net.layers(1:end-1);
sz=40;
stride=6;%floor(227/2);
str_nm=1;%floor(sz/2);

r=size(img,1);
c=size(img,2);
(floor((size(img,1)-227)/stride)+1);
(floor((size(img,2)-227)/stride)+1);

% nm=zeros(stride*(floor((size(img,1)-227)/stride)+1),stride*(floor((size(img,2)-227)/stride)+1),3);
% weight=zeros(size(nm));
nm_r=floor((size(img,1)-227)*str_nm/stride)+sz;
nm_c=floor((size(img,2)-227)*str_nm/stride)+sz;

nm=zeros(nm_r,nm_c,3);
weight=zeros(size(nm));
p=1;q=1;


for i=1:stride:(r-227)
    q=1;
    for j=1:stride:(c-227)
        
        pat= uint8(img(i+(0:226),j+(0:226),:));
        weight(p+(0:sz-1),q+(0:sz-1),:)=weight(p+(0:sz-1),q+(0:sz-1),:)+1;
        
        pat=gpuArray(single(pat-128));
        
        
        res=vl_simplenn(net,pat);
        res=res(end).x;
        res=reshape(res,[3 sz*sz]);
        res=res';
        res= (res'./repmat(sqrt(sum(abs(res').^2,1)),[3 1]))';
        res=reshape(res,[40 40 3]);
        
        nm(p+(0:sz-1),q+(0:sz-1),:)=nm(p+(0:sz-1),q+(0:sz-1),:)+gather(res);
        
        q=q+str_nm;
        
        
    end
    p=p+str_nm;
    
end

%weight(weight(:)==0)=1;
nm=nm./weight;
%nm=nm(1:p,1:q,:);
a=nm;
a(isnan(a))=0;
a=reshape(nm,[size(nm,1)*size(nm,2) 3]);
a=(a'./repmat(sqrt(sum(abs(a').^2,1)),[3 1]))';
a=reshape(a,size(nm));
nm=a;



b=figure();
set(b,'PaperUnits','inches','PaperPosition',[0 0 13.64 6.36])
%b.PaperPosition=[0 0 636 1364];
%normals=imresize(normals,[size(nm,1) size(nm,2)]);
% subplot(1,3,1);
% imagesc(imresize(img,[size(nm,1) size(nm,2)]));
subplot(1,2,1);
imagesc(cat(3,imadjust(mat2gray(nm(:,:,1))),imadjust(mat2gray(nm(:,:,2))),imadjust(mat2gray(nm(:,:,3)))));
subplot(1,2,2);
%imagesc(cat(3,imadjust(mat2gray(normals(:,:,1))),imadjust(mat2gray(normals(:,:,2))),imadjust(mat2gray(normals(:,:,3)))));
imagesc(imresize(img,[size(nm,1) size(nm,2)]));
%print(b,sprintf('Results/Combined_normal_maps/res%d',num),'-djpeg','-r100');
print(b,sprintf('Results/Combined_normal_maps/test%d',num),'-djpeg','-r100');