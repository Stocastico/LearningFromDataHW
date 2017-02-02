%% SPECIFY CONSTANTS
DIM = 2;
INTERVAL = [-1 1];

doPlots = false;
plotX = linspace(INTERVAL(1), INTERVAL(2), 100);
Ein = zeros(NUM_REPEAT, 1);
Eout = zeros(NUM_REPEAT, 1);
iter = zeros(NUM_REPEAT, 1);

%% READ DATA
fid = fopen('../in.dta');
in = fscanf(fid, '%f %f %f');
in = reshape(in, 3, length(in)/3);
in = in';
fclose(fid);

fid = fopen('../out.dta');
out = fscanf(fid, '%f %f %f');
out = reshape(out, 3, length(out)/3);
out = out';
fclose(fid);

%% APPLY NON LINEAR TRANSFORMATION
nTrain = size(in, 1);
inT = [ones(nTrain, 1), in(:, 1), in(:, 2), in(:,1).^2, in(:,2).^2, in(:,1) .* in(:,2).^2, ...
  abs(in(:,1) - in(:,2)), abs(in(:,1) + in(:,2))];

nTest = size(out, 1);
test = [ones(nTest, 1), out(:, 1), out(:, 2), out(:,1).^2, out(:,2).^2, out(:,1) .* out(:,2).^2, ...
  abs(out(:,1) - out(:,2)), abs(out(:,1) + out(:,2))];

%% QUESTION 1 AND 2
disp('QUESTION 1 AND 2')
train = inT(1:25, :);
valid = inT(26:end, :);

yTrain = in(1:25, 3);
yValid = in(26:end, 3);
yTest  = out(:, 3);

for kk = 4:8
  linReg = LinearRegression(kk);
  linReg = linReg.train(train(:, 1:kk), yTrain);
  
  % validation error
  yhat = linReg.classify(valid(:, 1:kk));
  E_valid = sum(yhat ~= yValid) / numel(yValid);
  disp(['Validation error for k = ' num2str(kk-1) ' : ' num2str(E_valid)])
  
  % out of sample error
  yhat = linReg.classify(test(:, 1:kk));
  E_test = sum(yhat ~= yTest) / numel(yTest);
  disp(['Test error for k = ' num2str(kk-1) ' : ' num2str(E_test)])
end

%% QUESTION 3 AND 4
disp('QUESTION 3 AND 4')
train = inT(26:end, :);
valid = inT(1:25, :);

yTrain = in(26:end, 3);
yValid = in(1:25, 3);
yTest  = out(:, 3);

for kk = 4:8
  linReg = LinearRegression(kk);
  linReg = linReg.train(train(:, 1:kk), yTrain);
  
  % validation error
  yhat = linReg.classify(valid(:, 1:kk));
  E_valid = sum(yhat ~= yValid) / numel(yValid);
  disp(['Validation error for k = ' num2str(kk-1) ' : ' num2str(E_valid)])
  
  % out of sample error
  yhat = linReg.classify(test(:, 1:kk));
  E_test = sum(yhat ~= yTest) / numel(yTest);
  disp(['Test error for k = ' num2str(kk-1) ' : ' num2str(E_test)])
end

%% QUESTION 8
NUMPTS = [10 100];
NUM_REPEAT = 1000;
E_in = zeros(length(NUMPTS), NUM_REPEAT);
E_out_pla = E_in;
E_out_svm = E_in;
numSV = E_in;
for jj = 1:length(NUMPTS);
  for kk = 1: NUM_REPEAT
    %% DEFINE A LINE
    linePts = INTERVAL(1) + (INTERVAL(2) - INTERVAL(1)) .* rand(2, 2);
    %lineequation =  A + Bx + Cy = 0; V = (A, B, C)
    V = [linePts(2,1)*linePts(1,2) - linePts(1,1)*linePts(2,2), linePts(2, 2) - linePts(1, 2), linePts(1, 1) - linePts(2, 1)];
    m = -V(2) / V(3);
    q = -V(1) / V(3);
    plotY = q + m*plotX;

    %% EXTRACT NUMPTS POINTS
    flag = true;
    while flag
      X = ones(NUMPTS(jj), 1+DIM);
      X(:, 2:1+DIM) = INTERVAL(1) + (INTERVAL(2) - INTERVAL(1)) .* rand(NUMPTS(jj), DIM);
      y = double(X(:,3) - m * X(:,2) - q > 0);
      y(y == 0) = -1;
      if ~(all(y == 1) || all (y == -1))
        flag = false;
      end
    end

    %% RUN PLA
    pla = PLA(1+DIM);
    pla = pla.train(X, y);
    yhat = pla.classify(X);
    E_in(jj, kk) = sum(yhat(:) ~= y) / numel(y);
    
    %% RUN SVM
    model = fitcsvm(X, y);
    numSV(jj, kk) = sum(model.IsSupportVector);
    
    %% TEST DATA
    Xtest = ones(10000, 1+DIM);
    Xtest(:, 2:1+DIM) = INTERVAL(1) + (INTERVAL(2) - INTERVAL(1)) .* rand(10000, DIM);
    ytest = double(Xtest(:, 3) - m * Xtest(:, 2) - q > 0);
    ytest(ytest == 0) = -1;
    yhat = pla.classify(Xtest);
    yhat2 = model.predict(Xtest);
    E_out_pla(jj, kk) = sum(yhat(:) ~= ytest) / numel(yhat);
    E_out_svm(jj, kk) = sum(yhat2 ~= ytest) / numel(yhat2);
  end
end