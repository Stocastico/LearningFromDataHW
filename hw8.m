%% SPECIFY CONSTANTS
DIM = 2;
INTERVAL = [-1 1];

%% READ DATA
fid = fopen('../features.train');
in = fscanf(fid, '%f %f %f');
in = reshape(in, 3, length(in)/3);
in = in';
fclose(fid);

fid = fopen('../features.test');
out = fscanf(fid, '%f %f %f');
out = reshape(out, 3, length(out)/3);
out = out';
fclose(fid);

%% QUESTION 2-4

C = 0.01;
Q = 2; %degree of polynomial

ex2 = {in; in; in; in; in; in; in; in; in; in};
vals = [0 1 2 3 4 5 6 7 8 9];
accuracy = zeros(1, length(vals));
for k = 1: length(vals)
  disp(['----- ' num2str(vals(k)) ' -----'])
  ex2{k}(ex2{k}(:,1) ~= vals(k)) = -1; 
  ex2{k}(ex2{k}(:,1) == vals(k)) = 1;
  model = svmtrain(ex2{k}(:,1), ex2{k}(:,2:3), '-s 0 -t 1 -d 2 -r 1 -c 0.01');
  [~, accuracy, ~] = svmpredict(ex2{k}(:,1), ex2{k}(:,2:3), model)
end

%% QUESTION 5-6
ex5 = in;
ex5(ex5(:,1) ~= 1 & ex5(:,1) ~= 5, :) = [];
ex5(ex5(:,1) == 5, 1) = -1;
C = [0.0001 0.001 0.01 0.1 1];
Q = [2 5];
for k = 1:length(C)
  for m = 1:length(Q)
    disp(['------ C = ' num2str(C(k)) ' Q = ' num2str(Q(m)) ' ------'])
    opts = ['-s 0 -t 1 -d ' num2str(Q(m)) ' -r 1 -c ' num2str(C(k))];
    model = svmtrain(ex5(:,1), ex5(:, 2:3), opts);
    [~, accuracy, ~] = svmpredict(ex5(:,1), ex5(:, 2:3), model)
  end
end

%% QUESTION 7-8
ex7 = ex5;
C = [0.0001 0.001 0.01 0.1 1];
Q = 2;
NUM_REPEAT = 100;
choice = zeros(1, 5);
accu = zeros(1, 5);
totalAccu = zeros(1, 5);
for m = 1:NUM_REPEAT
  disp(['------ Iteration = ' num2str(m) ' ------'])
  perm = randperm(length(ex7));
  tmp = ex7(perm, :);
  for k = 1:length(C)
    opts = ['-s 0 -t 1 -d ' num2str(Q) ' -v 10 -r 1 -c ' num2str(C(k))];
    model = svmtrain(tmp(:,1), tmp(:, 2:3), opts);
    accu(k) = model;
  end
  [~, idx] = max(accu);
  totalAccu = totalAccu + accu;
  choice(idx) = choice(idx) + 1;
end
choice
(100 - (totalAccu/100)) / 100;


%% QUESTION 9-10
ex9 = ex5;
ex9out = out;
ex9out(ex9out(:,1) ~= 1 & ex9out(:,1) ~= 5, :) = [];
ex9out(ex9out(:,1) == 5, 1) = -1;
C = [0.01 1 100 10000 1000000];
for k = 1:length(C)
  disp(['------ C = ' num2str(C(k)) ' ------'])
  opts = ['-s 0 -t 2 -g 1 -c ' num2str(C(k))];
  model = svmtrain(ex9(:,1), ex9(:, 2:3), opts);
  [~, accuracyIn, ~] = svmpredict(ex9(:,1), ex9(:, 2:3), model);
  [~, accuracyOut, ~] = svmpredict(ex9out(:,1), ex9out(:, 2:3), model);
end