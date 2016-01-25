function [res,label]=view_res(test,n)

fprintf('Using Provided Test Patch');
imdb=test;
npatches = n;

load('data/cnnNormal-baseline/net-epoch-200.mat');
run ../matconvnet-master/matlab/vl_setupnn.m;
load meanpatch.mat;
%load zeromean_unitvariance.mat;
load whiten_data_matrices.mat;

net.layers=net.layers(1:end-1);

da=imdb.images.data;
% da=single(imdb.images.data)-repmat(meanpatch,[1 1 1 size(imdb.images.data,4)]);
da=single(imdb.images.data(:,:,:,n))-128;
da=gpuArray(single(da));
sz=1600;

t=vl_simplenn(net,da);

t=t(end).x;
t=gather(t);
size(t)

% for i=npatches

    res=t;

    res=reshape(res,[3 sz]);
    
%     res=(res.*repmat(stddev,[1 size(res,2)]))+repmat(avg,[1 size(res,2)]);
%     res=res./repmat(sqrt(sum(abs(res).^2,1)),[3 1]);
%     res=res';
%    res=reshape(res,[40 40 3]);
%Xrec = Xwh*invMat + repmat(mu, size(X,1),1);
     res=res';
     %res= res*invMat +repmat(avg,[size(res,1) 1]) ;
     res= (res'./repmat(sqrt(sum(abs(res').^2,1)),[3 1]))';
     %res=res';
     res=reshape(res,[40 40 3]);


    %res=exp(res);
    a=figure(2);
    suptitle('lr=0.01  drprate=0.5  lrscale=500 scale =10 init_bias= 0.01 learningrate= 0.01 200 epochs ');
    subplot(1,7,1);


%     label=reshape(imdb.images.labels(:,:,:,i),[3 1600]);
%     label=(label.*repmat(stddev,[1 size(label,2)]))+repmat(avg,[1 size(label,2)]);
%     label=label./repmat(sqrt(sum(abs(label).^2,1)),[3 1]);
%     label=label';
%     label=reshape(label,[40 40 3]);

    label=reshape(imdb.images.labels(:,:,:,n),[3 1600]);
    label=label';
    %label=label*invMat +repmat(avg,[size(label,1) 1]);
    label= (label'./repmat(sqrt(sum(abs(label').^2,1)),[3 1]))';
    label=reshape(label,[40 40 3]);
    label(isnan(label))=0;

    
 
    
    %imagesc(cat(3,imadjust(mat2gray(label(:,:,1))),imadjust(mat2gray(label(:,:,2))),ones(40,40)));
    imagesc(imadjust(mat2gray(label(:,:,1))));
    subplot(1,7,2);
    
    %imagesc(cat(3,imadjust(mat2gray(res(:,:,1))),imadjust(mat2gray(res(:,:,2))),imadjust(mat2gray(ones(40,40)))));
     %imagesc(cat(3,imadjust(mat2gray(res(:,:,1))),imadjust(mat2gray(res(:,:,2))),ones(40,40)));
    imagesc(imadjust(mat2gray(res(:,:,1))));
    
    subplot(1,7,3);
    imagesc(imadjust(mat2gray(label(:,:,2))));
    subplot(1,7,4);
    imagesc(imadjust(mat2gray(res(:,:,2))));
    %print(a,sprintf('Results/image%d/img_%d.jpg',n,i),'-djpeg');
    %close(a)
    subplot(1,7,5);
    imagesc(cat(3,imadjust(mat2gray(label(:,:,1))),imadjust(mat2gray(label(:,:,2))),imadjust(mat2gray(label(:,:,3)))));
    
    subplot(1,7,6);
    imagesc(cat(3,imadjust(mat2gray(res(:,:,1))),imadjust(mat2gray(res(:,:,2))),imadjust(mat2gray(res(:,:,3)))));
    ff=1
    subplot(1,7,7);
    imagesc(imdb.images.data(:,:,:,n));
% end

% for i=npatches
% 
%     res=t(:,:,:,i);
%     
%      res=reshape(res,[40 40 2]);
% %    res=reshape(res,[40 40]);
%     res=exp(res);
%     %res(:,:,1)=res(:,:,1)*2*pi;
%     %res(:,:,2)=res(:,:,2)*pi;
%     
%     %a=figure('Visible','off');
%     a=figure(2)
%     suptitle('lr=0.001  drprate=0.5  lrscale=20 learningrate= 0.1 400 epochs ');
%     subplot(1,4,1);
% 
% %     label=reshape(imdb.images.labels(:,:,:,i),[40 40 2]);
%     label=reshape(imdb.images.labels(:,:,:,i),[40 40 2]);
% 
%     
%     %label(:,:,1)=label(:,:,1)*2*pi;
%     %label(:,:,2)=label(:,:,2)*pi;
%     
%     %imagesc(cat(3,imadjust(mat2gray(label(:,:,1))),imadjust(mat2gray(label(:,:,2))),ones(40,40)));
%     imagesc(imadjust(mat2gray(label(:,:,1))));
%     subplot(1,4,2);
%     
%     %imagesc(cat(3,imadjust(mat2gray(res(:,:,1))),imadjust(mat2gray(res(:,:,2))),imadjust(mat2gray(ones(40,40)))));
%      %imagesc(cat(3,imadjust(mat2gray(res(:,:,1))),imadjust(mat2gray(res(:,:,2))),ones(40,40)));
%     imagesc(imadjust(mat2gray(res(:,:,1))));
%     
%     subplot(1,4,3);
%     imagesc(imadjust(mat2gray(label(:,:,2))));
%     subplot(1,4,4);
%     imagesc(imadjust(mat2gray(res(:,:,2))));
%     %print(a,sprintf('Results/image%d/img_%d.jpg',n,i),'-djpeg');
%     %close(a)
%     ff=1
% end
