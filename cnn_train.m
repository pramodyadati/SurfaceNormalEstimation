%---------------------------------------------------------------------------------------------------------------------
% CNN_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastic gradient
%    descent with momentum to train a CNN for image classification.
%    It can be used with different datasets by providing a suitable
%    getBatch function.
%---------------------------------------------------------------------------------------------------------------------

function [net, info] = cnn_train(net, imdb, getBatch,num,conc,varargin)


opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 300 ;
opts.batchSize = 256 ;
opts.useGpu = true ;
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.sync = true ;
opts.prefetch = false ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
%opts.errorType = 'multiclass' ;
opts.plotDiagnostics = false ;
opts = vl_argparse(opts, varargin) ;
if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

for i=1:numel(net.layers)
  if ~strcmp(net.layers{i}.type,'conv'), continue; end
  net.layers{i}.filtersMomentum = zeros(size(net.layers{i}.filters), ...
    class(net.layers{i}.filters)) ;
  net.layers{i}.biasesMomentum = zeros(size(net.layers{i}.biases), ...
    class(net.layers{i}.biases)) ; %#ok<*ZEROLIKE>
  if ~isfield(net.layers{i}, 'filtersLearningRate')
    net.layers{i}.filtersLearningRate = 1 ;
  end
  if ~isfield(net.layers{i}, 'biasesLearningRate')
    net.layers{i}.biasesLearningRate = 1 ;
  end
  if ~isfield(net.layers{i}, 'filtersWeightDecay')
    net.layers{i}.filtersWeightDecay = 1 ;
  end
  if ~isfield(net.layers{i}, 'biasesWeightDecay')
    net.layers{i}.biasesWeightDecay = 1 ;
  end
end

if opts.useGpu
  net = vl_simplenn_move(net, 'gpu') ;
  for i=1:numel(net.layers)
    if ~strcmp(net.layers{i}.type,'conv'), continue; end
    net.layers{i}.filtersMomentum = gpuArray(net.layers{i}.filtersMomentum) ;
    net.layers{i}.biasesMomentum = gpuArray(net.layers{i}.biasesMomentum) ;
  end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

rng(0) ;

if opts.useGpu
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end

info.train.objective = [] ;
info.train.error = [] ;
info.train.topFiveError = [] ;
info.train.speed = [] ;
info.val.objective = [] ;
info.val.error = [] ;
info.val.topFiveError = [] ;
info.val.speed = [] ;

lr = 0 ;
res = [] ;

for epoch=1:opts.numEpochs
    
    prevLr = lr ;
    lr = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    
    % fast-forward to where we stopped
    
     modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
     modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
    
%     if opts.continue
%         if exist(modelPath(epoch),'file')
%             if epoch == opts.numEpochs
%                 load(modelPath(epoch), 'net', 'info') ;
%             end
%             continue ;
%         end
%         if epoch > 1
%             fprintf('resuming by loading epoch %d\n', epoch-1) ;
%             load(modelPath(epoch-1), 'net', 'info') ;
%         end
%     end

    train = opts.train(randperm(numel(opts.train))) ;
    val = opts.val ;
    
    info.train.objective(end+1) = 0 ;
    info.train.error(end+1) = 0 ;
    info.train.topFiveError(end+1) = 0 ;
    info.train.speed(end+1) = 0 ;
    info.val.objective(end+1) = 0 ;
    info.val.error(end+1) = 0 ;
    info.val.topFiveError(end+1) = 0 ;
    info.val.speed(end+1) = 0 ;

    % reset momentum if needed
    if prevLr ~= lr
        fprintf('learning rate changed (%f --> %f): resetting momentum\n', prevLr, lr) ;
        for l=1:numel(net.layers)
            if ~strcmp(net.layers{l}.type, 'conv'), continue ; end
            net.layers{l}.filtersMomentum = 0 * net.layers{l}.filtersMomentum ;
            net.layers{l}.biasesMomentum = 0 * net.layers{l}.biasesMomentum ;
        end
    end
    
    for t=1:opts.batchSize:numel(train)
        %------------------------------------
        % get next image batch and labels
        %------------------------------------
        
        batch = train(t:min(t+opts.batchSize-1, numel(train))) ;
        batch_time = tic ;
        fprintf('training: epoch %02d: processing batch %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize)) ;
        
        [im, labels] = getBatch(imdb,batch) ;
        
        if opts.prefetch
            nextBatch = train(t+opts.batchSize:min(t+2*opts.batchSize-1, numel(train))) ;
            getBatch(imdb,nextBatch) ;
        end
        if opts.useGpu
            im = gpuArray(im) ;
        end
        %------------------------------------
        % backprop
        %------------------------------------
    
        net.layers{end}.class = labels ;
        net.layers{end}.w = labels ;
        if (conc==1)
            res=vl_simplennConcat(net, im, one, res, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync) ;
        else
            res = vl_simplenn(net, im, one, res, ...
            'conserveMemory', opts.conserveMemory, ...
            'sync', opts.sync) ;
        end
        %------------------------------------
        % gradient step
        %------------------------------------
        
        for l=1:numel(net.layers)
            if ~strcmp(net.layers{l}.type, 'conv'), continue ; end

            net.layers{l}.filtersMomentum = ...
            opts.momentum * net.layers{l}.filtersMomentum ...
            - (lr * net.layers{l}.filtersLearningRate) * ...
            (opts.weightDecay * net.layers{l}.filtersWeightDecay) * net.layers{l}.filters ...
            - (lr * net.layers{l}.filtersLearningRate) / numel(batch) * res(l).dzdw{1} ;

            net.layers{l}.biasesMomentum = ...
            opts.momentum * net.layers{l}.biasesMomentum ...
            - (lr * net.layers{l}.biasesLearningRate) * ....
            (opts.weightDecay * net.layers{l}.biasesWeightDecay) * net.layers{l}.biases ...
            - (lr * net.layers{l}.biasesLearningRate) / numel(batch) * res(l).dzdw{2} ;

            net.layers{l}.filters = net.layers{l}.filters + net.layers{l}.filtersMomentum ;
            net.layers{l}.biases = net.layers{l}.biases + net.layers{l}.biasesMomentum ;
        end

        %------------------------------------
        % print information
        %------------------------------------
        
        batch_time = toc(batch_time) ;
        speed = numel(batch)/batch_time ;
        info.train = updateError(opts, info.train, net, res, batch_time) ;

        fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;

        n = t + numel(batch) - 1 ;
        %fprintf(' err %.5f ', ...
        %info.train.error(end)) ;
        %fprintf('\n') ;

        % debug info
        if opts.plotDiagnostics
            figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
        end
    end % next batch

    
    % save
    info.train.objective(end) = info.train.objective(end)/(ceil(numel(train)/opts.batchSize));% / numel(train) ;
    info.train.error(end) = info.train.error(end) / numel(train)  ;
    info.train.topFiveError(end) = info.train.topFiveError(end) / numel(train) ;
    info.train.speed(end) = numel(train) / info.train.speed(end) ;
    info.val.objective(end) = info.val.objective(end) / numel(val) ;
    info.val.error(end) = info.val.error(end) / numel(val) ;
    info.val.topFiveError(end) = info.val.topFiveError(end) / numel(val) ;
    info.val.speed(end) = numel(val) / info.val.speed(end) ;
    %lr=opts.train.learningRate;
    if( epoch==200 || epoch==300 ||epoch==400 )
       save(modelPath(epoch), 'net', 'info') ;
    end

    figure(1) ; clf ;
    %subplot(1,2,1) ;
    semilogy(1:epoch, info.train.objective, 'k') ; hold on ;
    xlabel('Training epoch') ; ylabel('Energy (Log Error)') ;
    grid on ;
    h=legend('train') ;%, 'val') ;
    set(h,'color','none');
    title('objective( Logarithmic Scale)') ;
    
    figure(2) ; clf ;
    plot(1:epoch, info.train.objective, 'b') ; hold on ;
    % semilogy(1:epoch, info.val.objective, 'b') ;
    xlabel('Training epoch') ; ylabel('Average Error') ;
    grid on ;
    h=legend('train') ;%, 'val') ;
    set(h,'color','none');
    title('objective') ;
    
    drawnow ;
    out = 1
    %figure(2); clf;
    %view_res(net,imdb);

%    print(1, modelFigPath, '-dpdf') ;
%    clear imdb;
%    imdb=randpatnorm(num);
    
end

% -------------------------------------------------------------------------
function info = updateError(opts, info, net, res, speed)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
sz = size(predictions) ;
n = prod(sz(1:2)) ;

labels = net.layers{end}.class ;
% fprintf('\n\n Batch Error: %f  Angle error (Inv Cos) : %f\n',res(end).x,acosd(1-gather(res(end).x/2)));
 fprintf('\n\n Batch Error: %f  Angle error (Inv Cos) : %f\n',res(end).x,acosd(-gather(res(end).x)));
%fprintf('\n\n Batch Error: %f  \n',res(end).x);

info.objective(end) = info.objective(end) + (sum(double(gather(squeeze(res(end).x))))) ;

%fprintf('\n Objective Function value: %f\n\n',info.objective(end));
info.speed(end) = info.speed(end) + speed ;
info.error(end) = 0;



