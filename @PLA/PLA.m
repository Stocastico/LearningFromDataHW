classdef PLA
  %PLA Perceptron Learning Algorithm
  %   Simple implementation of the perceptron learning algorithm
  
  properties
    pocket = false; % if true, use the pocket algorithm to store the best current solution
    w;              % weights
    N;              % dimensionality (including the bias)
    maxIt = 10000;  % maximum number of iterations
    numIt;          % number of iterations until convergence
  end
  
  methods
    function obj = PLA(nDim, w, doPocket)
    %PLA constructor of PLA object
      obj.N = nDim;
      if nargin > 1
        obj.w = w;
      else
        obj.w = zeros(obj.N, 1);
      end
      if nargin > 2
        obj.pocket = doPocket;
      end
    end
    
    function obj = train(obj, X, y)
    %TRAIN performs training
      if iscolumn(y)
        y = y';
      end
      obj.maxIt = size(X, 2);
      classif = sign(obj.w' * X');
      correct = isequal(y, classif);
      obj.numIt = 0;
      NUMPTS = length(y);
      while ~correct && obj.numIt < obj.maxIt
        %update number of iterations
        obj.numIt = obj.numIt + 1;
        %choose a wrongly assigned point
        posPt = randi(NUMPTS);
        while classif(posPt) == y(posPt)
          posPt = randi(NUMPTS);
        end
        %update weights
        obj.w = obj.w + (y(posPt) * X(posPt, :))';
        classif = sign(obj.w' * X');
        correct = isequal(y, classif);
      end
    end
    
    function y = classify(obj, x)
    %CLASSIFY classify a new sample
      y = sign(obj.w' * x');
    end
  end
  
end

