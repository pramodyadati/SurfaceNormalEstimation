function imdb=randpatnorm(num,varargin)

%load Similarly_Illuminated.mat;
%load whiten_data_matrices_20.mat;
%load Better_Surfnorm.mat;
load whitened_surfnorm.mat;

noi=size(imgData,2);
nop=ceil(num/noi);
%nop=num;
np=1;
np_img=1;


data =zeros(227,227,3,num,'uint8');
labels=zeros(1,1,4800,num,'single');
%orig=zeros(1,1,1200,num,'single');

for i=1:noi


	r=size(imgData(i).img,1);
    c=size(imgData(i).img,2);
    img=imgData(i).img;
    
	nm=imgData(i).nm;
    if nargin>1
        randx=95; %45
        randy=95; %45
    else    
        randx=randi(r-227+1,nop,1);
        randy=randi(c-227+1,nop,1);
    end

	for j=1:nop
	
		data(:,:,:,np)=uint8(img(randx(j)+(0:226),randy(j)+(0:226),:));

		a=imresize(single(nm(randx(j)+(0:226),randy(j)+(0:226),:)),[40 40]);
        %temp=a(:,:,1);
        %a(:,:,1)=(a(:,:,1)-mean(temp(:)))/(std(temp(:)));
        %a(:,:,1)=(a(:,:,1)-mean_t)/(std_t);
        %temp=a(:,:,2);
        %a(:,:,2)=(a(:,:,2)-mean(temp(:)))/(std(temp(:)));
        %a(:,:,2)=(a(:,:,2)-mean_p)/(std_p);

        a=reshape(a,[1600 3]);
        %a_1=(a-repmat(avg, size(a,1),1))*whMat;    % Whitening Surface Normals
        %a_1=single(a');
        a=a';
        a(isnan(a))=0;
        % Zero mean and Unit variance of x,y,z
        
        %a=(a-repmat(avg,[1 size(a,2)]))./(repmat(stddev,[1 size(a,2)]));
        
        labels(:,:,:,np)=(reshape(a,[1 1 4800]));
        %orig(:,:,:,np)=(reshape(a,[1 1 1200]));
        
        np=np+1;
		j;
        i;
		if(np>num)
			break;
		end
	end
    np;
	i;
	
    if(np>num)
		break;
	end
	
end

clear imgData;

imdb.images.data=zeros(227,227,3,num,'uint8');
imdb.images.labels=zeros(1,1,4800,num,'single');
%imdb.images.orig=zeros(1,1,1200,num,'single');



a=randperm(num);
for i=1:num
imdb.images.data(:,:,:,i)=uint8(data(:,:,:,a(i)));
end
clear data;
for i=1:num
%     temp=reshape(labels(:,:,:,a(i)),[40 40 2]);
%     %temp=squeeze(temp(:,:,1));fprintf('Just using channel 1')
%     temp=squeeze(temp(:,:,1));fprintf('Using Both Channels')
    
imdb.images.labels(:,:,:,i)=labels(:,:,:,a(i));
%imdb.images.orig(:,:,:,i)=orig(:,:,:,a(i));
end

clear labels a randx randy np j nop ;

imdb.images.set=ones(1,1,num);
imdb.meta.sets = {'train', 'val', 'test'} ;
%save(sprintf('NormData/oneimg_%d.mat',noi),'imdb');