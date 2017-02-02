classdef LinearRegression
  %LINEARREGRESSION Linear regression Learning Algorithm
  %   Simple implementation of the linear regression learning algorithm
  
  properties
    w;              % weights
    N;              % dimensionality (including the bias)
    lambda = 0;     % regularization factor
  end
  
  methods
    function obj = LinearRegression(nDim, lambda)
    %LinearModel constructor of LinearModel object
      obj.N = nDim;
      obj.w = zeros(1, obj.N);
      if nargin > 1
        obj.lambda = lambda;
      end
    end
    
    function obj = train(obj, X, y)
    %TRAIN performs training
      
      %check that y is a column vector
      if isrow(y)
        y = y';
      end
      if 0 == obj.lambda
        obj.w = pinv(X) * y;
      else
        %obj.w = inv(X' * X + obj.lambda * eye(length(obj.w))) * X' * y;
        obj.w = (X' * X + obj.lambda * eye(length(obj.w))) \ X' * y;
      end
    end
    
    function y = predict(obj, x)
    %PREDICT  predict new samples
      y = obj.w' * x';
      if 1 == size(y, 1)
        y = y';
      end
    end
    
    function y = classify(obj, x)
    %CLASSIFY classify new samples
      y = sign(predict(obj, x));
    end
  end
  
end

