classdef LogisticRegression
  %LOGISTICREGRESSION Logistic Regression algorithm
  %   Simple implementation of the logistic regression learning algorithm
  
  properties
     w;                   % weights
     N;                   % dimensionality (including the bias)
     learn_rate = 0.01;   % learning rate
     batch_size = 1;      % size of batches passed to SGD
     max_epoch = 1000;    % maximum number of iteration before stopping training
     max_err = 0.01;      % maximum error before stopping the training
     momentum = 0.9;      % momentum used when doing SGD
     sigmoid;             % function computing the sigmoid
     sigmoid_deriv;       % function computing the derivative of sigmoid
  end
  
  methods
    function obj = LogisticRegression(nDim)
    %LinearModel constructor of LinearModel object
      obj.N = nDim;
      obj.w = zeros(1, obj.N);
      obj.sigmoid = @(x) ( 1 ./ (1 + exp(-x)));
      obj.sigmoid_deriv = @(x)( obj.sigmoid(x) .* (1 - obj.sigmoid(x) ));
    end
    
    function [obj, epoch] = trainEx8(obj, X, y)
    
      [nSamples, ~] = size(X);
      epoch = 0;
      deltaW = inf;
      deltaWThresh = 0.01;
      
      while deltaW > deltaWThresh
        currW = obj.w;
        epoch = epoch + 1;
        order = randperm(nSamples);
        Xp = X(order, :);
        yp = y(order);
        for k = 1:nSamples
          obj.w = obj.w - obj.learn_rate * obj.errorGradient(Xp(k, :), yp(k));
        end
        deltaW = norm(obj.w - currW);
        E_curr = obj.calcError(Xp, yp);
        %if 0 == mod(epoch, 50), disp(['Epoch: ' num2str(epoch) ' E_in = ' num2str(E_curr)]); end
      end
      
    end
    
    function y = predict(obj, x)
    %PREDICT  predict new samples
      y = obj.sigmoid(obj.w * x');
      if 1 == size(y, 1)
        y = y';
      end
    end
    
    function err = calcError(obj, X, y)
    %CALCERROR Computes the cross-entropy error
      
      [nSamples, ~] = size(X);
      temp = 0;
      for k = 1:nSamples
        temp = temp + log(1 + exp(-y(k) * obj.w * X(k, :)') );
      end
      err = temp / nSamples;
    end
    
    function errGrad = errorGradient(obj, X, y)
    %ERRORGRADIENT computes the error gradient
    
      [nSamples, ~] = size(X);
      errGrad = 0;
      for k = 1:nSamples
        errGrad = errGrad + (y(k) * X(k, :))/(1 + exp(y(k) * obj.w * X(k, :)'));
      end
      errGrad = errGrad / (-nSamples);
    end
    

  end
  
end

