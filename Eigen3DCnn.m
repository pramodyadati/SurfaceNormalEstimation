function imdb=Eigen3DCnn(num)
DOUBLETHETA = false;
%-----------------------------------------------------------------------------------------------------------------
% Setup matconvnet and load Pretrained Imagenet Network
%-----------------------------------------------------------------------------------------------------------------
run ../matconvnet-master/matlab/vl_setupnn ;
%run /home/cvit/matconvnet-master_node14/matlab/vl_setupnn;
load('eigen_converted.mat');

%-----------------------------------------------------------------------------------------------------------------
% Initialization
%-----------------------------------------------------------------------------------------------------------------

opts.dataDir = fullfile('data','cnnDepth') ;
opts.expDir = fullfile('data','cnnNormal-baseline') ;
opts.imdbPath = fullfile('data', 'imdb_l.mat');
opts.train.batchSize = 32;
opts.train.numEpochs = 200 ;
opts.train.continue = true ;
opts.train.useGpu = true;
trigger_t = 120;
%opts.train.learningRate = [0.1*ones(1,trigger_t) 0.01*ones(1,200-trigger_t) ];
opts.train.learningRate = [0.01*ones(1,opts.train.numEpochs) ];
%opts.train.learningRate = logspace(-2,-3,opts.train.numEpochs);
%opts.train.learningRate= [0.1*ones(1,20), 0.01*ones(1,120), 0.001*ones(1,60)]
%opts.train.learningRate = cat(2,0.1*ones(1,20),logspace(-2,-4,50),logspace(-4,-6,130));
opts.train.expDir = opts.expDir ;


%------------------------------------------------------------------------------------------------------------------------------
% Learning  Rate, Dropout Rate, Learning rate scale values
%-----------------------------------------------------------------------------------------------------------------------------
conc=0;
lr=0.1;
drprate=0.5;
lrscale=1000;
imdb=randpatnorm(num);
scaldwn_w=1;
scaldwn_b=1;


%------------------------------------------------------------------------------------------------------------------------------
% Network Architecture - Coarse Network- 5 layers of Alexnet and 2 new Fully
% Connected layers 
%------------------------------------------------------------------------------------------------------------------------------

scal = 100;
init_bias = 0.01;

net.layers = {} ;

% Block 1
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', layers(1).w, ...                         
                           'biases', layers(1).b, ...                          
                           'stride', 4, ...                                        
                           'pad', 0, ...
                           'filtersLearningRate', lr, ...
                           'biasesLearningRate', lr, ...
                           'filtersWeightDecay', 0.0005, ...
                           'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
                       
net.layers{end+1} = struct('type', 'normalize', ...
                           'param', [5 1 0.0001/5 0.75]) ;

% Block 2
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', layers(2).w, ...                         
                           'biases', layers(2).b, ...                          
                           'stride', 1, ...                                         
                           'pad', 2, ...
                           'filtersLearningRate',lr, ...
                           'biasesLearningRate', lr, ...
                           'filtersWeightDecay', 0.0005, ...
                           'biasesWeightDecay', 0) ;

net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;                       
net.layers{end+1} = struct('type', 'normalize', ...
                           'param', [5 1 0.0001/5 0.75]) ;


% Block 3
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', layers(3).w, ...                                             
                           'biases', layers(3).b, ...                           
                           'stride', 1, ...
                           'pad', 1, ...
                           'filtersLearningRate', lr, ...
                           'biasesLearningRate', lr, ...
                           'filtersWeightDecay', 0.0005, ...
                           'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 4
net.layers{end+1} = struct('type', 'conv', ...
                           'filters',layers(4).w, ...                            
                           'biases', layers(4).b, ...                            
                           'stride', 1, ...
                           'pad', 1, ...
                           'filtersLearningRate',lr, ...
                           'biasesLearningRate', lr, ...
                           'filtersWeightDecay', 0.0005, ...
                           'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'relu') ;

% Block 5
net.layers{end+1} = struct('type', 'conv', ...
                           'filters',layers(5).w, ...                            
                           'biases', layers(5).b , ...                            
                           'stride', 1, ...
                           'pad', 1, ...
                           'filtersLearningRate', lr, ...
                           'biasesLearningRate', lr, ...
                           'filtersWeightDecay', 0.0005, ...
                           'biasesWeightDecay', 0) ;

net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
% Block 6                                                                       
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(6,6,256,4096,'single') ,...                            
                           'biases', init_bias*ones(1,4096,'single') , ...                            
                           'stride', 1, ...
                           'pad', 0, ...
                           'filtersLearningRate', lr*lrscale, ...
                           'biasesLearningRate', lr*lrscale, ...
                           'filtersWeightDecay', 0.0005, ...
                           'biasesWeightDecay', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', ...
                           'rate', drprate) ;

% Block 7
if DOUBLETHETA
    net.layers{end+1} = struct('type', 'conv', ...
                           'filters', 0.01/scal * randn(1,1,4096,3200,'single'), ...
                           'biases', zeros(1, 3200, 'single'), ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'filtersLearningRate',lr*lrscale, ...
                           'biasesLearningRate', lr*lrscale, ...
                           'filtersWeightDecay', 0.00005, ...
                           'biasesWeightDecay', 0) ;

else
    net.layers{end+1} = struct('type', 'conv', ...
                               'filters', 0.01/scal * randn(1,1,4096,4800,'single'), ...
                               'biases', zeros(1,4800, 'single'), ...
                               'stride', 1, ...
                               'pad', 0, ...
                               'filtersLearningRate',lr*lrscale, ...
                               'biasesLearningRate', lr*lrscale, ...
                               'filtersWeightDecay', 0.00005, ...
                               'biasesWeightDecay', 0) ;
end
%net.layers{end+1} = struct('type','tanh');
% Block 9

     ly.type = 'custom' ;
    ly.forward = @nnNormdis_forward ;
    ly.backward = @nnNormdis_backward ;
%        ly.forward = @nndistance_forward ;
%        ly.backward = @nndistance_backward ;
%      
 net.layers{end+1} = ly ;
%net.layers{end+1} = struct('type', '3Dloss');



fprintf('\n Architecture Loaded..!!');

%---------------------------------------------------------------------------------------------------------------------------
% Training Process
%---------------------------------------------------------------------------------------------------------------------------

fprintf('\n Training Process underway..!! ');

[net, info] = cnn_train(net, imdb, @getBatch,num,conc, ...
                        opts.train, ...
                        'val', find(imdb.images.set == 3));
                        

fprintf('\n Training Done..!!');

%---------------------------------------------------------------------------------------------------------------------------
% GetBatch Function for batch processing 
%---------------------------------------------------------------------------------------------------------------------------

function [im, labels] = getBatch(imdb,batch)
 
    load('meanpatch.mat');
    %im = gpuArray(single(imdb.images.data(:,:,:,batch))-repmat(single(meanpatch),[1 1 1 size(batch,1)])) ;
    gpuArray(single(imdb.images.data(:,:,:,batch)) -128);
    %load avgpatch.mat;
    im = gpuArray(single(imdb.images.data(:,:,:,batch)));
    %im = gpuArray(im-repmat(avg,[1 1 1 size(im,4)]));

    %labels = log(single(imdb.images.labels(:,:,:,batch))) ; fprintf('Doing Log of Labels')
    labels = single(imdb.images.labels(:,:,:,batch)) ; fprintf('NoLog')
    %labels_rand = rand(size(labels));
    %  labels = repmat(labels,[1 1 2]);
    %  labels(:,:,3200:-1:1601) = labels(:,:,1:1600);
    out=1

    %labels = single(imdb.images.labels(:,:,:,batch)) ;
    %keyboard;
%--------------------------------------------------------------------------------------------------------------------------
