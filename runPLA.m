%% SPECIFY CONSTANTS
DIM = 2;
INTERVAL = [-1, 1];
NUMPTS = 100;
NUM_REPEAT = 1000;
numIt = zeros(1, NUM_REPEAT);
disag = zeros(1, NUM_REPEAT);
doPlots = false;
doErrorEstim = false;
plotX = linspace(INTERVAL(1), INTERVAL(2), 100);

for kk = 1:NUM_REPEAT

  %% DEFINE A LINE
  linePts = INTERVAL(1) + (INTERVAL(2) - INTERVAL(1)) .* rand(2, 2);
  %lineequation =  A + Bx + Cy = 0; V = (A, B, C)
  V = [linePts(2,1)*linePts(1,2) - linePts(1,1)*linePts(2,2), linePts(2, 2) - linePts(1, 2), linePts(1, 1) - linePts(2, 1)];
  m = -V(2) / V(3);
  q = -V(1) / V(3);
  plotY = q + m*plotX;

  %% EXTRACT NUMPTS POINTS
  X = ones(1+DIM, NUMPTS);
  X(2:1+DIM, :) = INTERVAL(1) + (INTERVAL(2) - INTERVAL(1)) .* rand(DIM, NUMPTS);
  y = double(X(3, :) - m * X(2, :) - q > 0);
  y(y == 0) = -1;
  
  %% PLOT DATASET
  if doPlots
    figure(1)
    gscatter(X(1,:), X(2,:), y)
    hold on
    plot(plotX, plotY, 'k');
    axis([INTERVAL(1) INTERVAL(2) INTERVAL(1) INTERVAL(2)])
  end

  %% RUN PLA
  pla = PLA(1+DIM);
  pla = pla.train(X, y);
  
  numIt(kk) = pla.numIt;
  
  %% ESTIMATE FUNCTION DISAGREEMENT
  if doErrorEstim
    %intersection of g vs f
    A = [V(2:3); (W(2:3))'];
    B = [-V(1); -W(1)];
    intersection = linsolve(A, B);
    ptX = intersection(1);
    ptY = intersection(2);

    %intersection of f with y = 0
    A = [V(2:3); 0, 1];
    B = [-V(1); INTERVAL(1)];
    int_F_minus1 = linsolve(A, B);

    %intersection of f with y = 2
    A = [V(2:3); 0, 1];
    B = [-V(1); INTERVAL(2)];
    int_F_plus1 = linsolve(A, B);
    int_F = sort([int_F_plus1(1), int_F_minus1(1)]);

    %intersection of g with y = 0
    A = [W(2:3)'; 0, 1];
    B = [-W(1); INTERVAL(1)];
    int_G_minus1 = linsolve(A, B);

    %intersection of g with y = 2
    A = [W(2:3)'; 0, 1];
    B = [-W(1); INTERVAL(2)];
    int_G_plus1 = linsolve(A, B);
    int_G = sort([int_G_plus1(1), int_G_minus1(1)]);

    startF = max(INTERVAL(1), int_F(1));
    endF = min(INTERVAL(2), int_F(2));
    startG = max(INTERVAL(1), int_G(1));
    endG = min(INTERVAL(2), int_G(2));
    addMeF = 0;
    addMeG = 0;
    if endF < INTERVAL(2)
      addMeF = 2 * (INTERVAL(2) - endF); 
    end
    if endG < INTERVAL(2)
      addMeG = 2 * (INTERVAL(2) - endG);
    end

    if ptX > INTERVAL(1) && ptX < INTERVAL(2) && ptY > INTERVAL(1) && ptY < INTERVAL(2)
      tmpX1F = linspace(startF, ptX, 100);
      tmpX2F = linspace(ptX, endF, 100);
      tmpX1G = linspace(startG, ptX, 100);
      tmpX2G = linspace(ptX, endG, 100);
      tmpF1 = m*tmpX1F + q;
      tmpF2 = m*tmpX2F + q;
      tmpG1 = (-W(1) / W(3)) + (-W(2) / W(3))*tmpX1G;
      tmpG2 = (-W(1) / W(3)) + (-W(2) / W(3))*tmpX2G;
      areaF1 = trapz(tmpX1F, tmpF1) / 4;
      areaF2 = (addMeF + trapz(tmpX2F, tmpF2)) / 4;
      areaG1 = trapz(tmpX1G, tmpG1) / 4;
      areaG2 = (addMeG + trapz(tmpX2G, tmpG2)) / 4;
      differ = (abs(areaF1 - areaG1) + abs(areaF2 - areaG2));
      if differ > 1
        differ = 0;
      end
      disag(kk) = differ;
    else
      tmpXF = linspace(startF, endF, 200);
      tmpXG = linspace(startG, endG, 200);
      tmpF = m*tmpXF + q;
      tmpG = (-W(1) / W(3)) + (-W(2) / W(3))*tmpXG;
      areaF = trapz(tmpXF, tmpF) / 4;
      areaG = trapz(tmpXG, tmpG) / 4;
      differ = abs(areaF - areaG);
      if differ > 1
        differ = 0;
      end
      disag(kk) = differ;
    end
  end
  
  if doPlots
    close(1)
  end
end

disp(mean(numIt));
disp(mean(disag));