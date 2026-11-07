%%======================================================
%% STEP 1a: Generate data from two 2D distributions.
% 從兩個 2D 高斯分佈產生資料點

mu1 = [1 2];       % Mean，第一個高斯分佈的平均向量
sigma1 = [ 3 .2;   % Covariance matrix，第一個高斯分佈的共變異數矩陣
          .2  2];
m1 = 200;          % Number of data points，第一群資料點的數量

mu2 = [-1 -2];     % 第二個高斯分佈的平均向量
sigma2 = [2 0;     % 第二個高斯分佈的共變異數矩陣
          0 1];
m2 = 100;          % 第二群資料點的數量

% Generate sample points with the specified means and covariance matrices.
R1 = chol(sigma1);      % 用 chol 把 covariance 拆掉，等一下 randn 產生的資料就會帶有 sigma 的形狀
X1 = randn(m1, 2) * R1; % 先產生 m1 個 2 維標準常態，再乘上 R1 加入相關性
X1 = X1 + repmat(mu1, size(X1, 1), 1); % 把整部分往 mu1 移動

R2 = chol(sigma2);      % 第二群，與上面第一群一樣
X2 = randn(m2, 2) * R2;
X2 = X2 + repmat(mu2, size(X2, 1), 1);

X = [X1; X2];           % 把兩群資料接起來變成同一個 Dataset

%%=====================================================
%% STEP 1b: Plot the data points and their pdfs.
% 把剛剛生的兩群資料畫出來，把它們原本真正的高斯分佈等高線畫出來。
figure(1);

% Display a scatter plot of the two distributions
hold off;
plot(X1(:, 1), X1(:, 2), 'bo'); % 第一群點
hold on;
plot(X2(:, 1), X2(:, 2), 'ro'); % 第二群點

set(gcf,'color','white') % White background for the figure. 背景白色

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
% 建一個 100x100 的網格，用來算 pdf
gridSize = 100;
u = linspace(-6, 6, gridSize);
[A B] = meshgrid(u, u);
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
% 算網格上每個座標在兩個高斯分佈下的 pdf 值
z1 = gaussianND(gridX, mu1, sigma1);
z2 = gaussianND(gridX, mu2, sigma2);

% Reshape the responses back into a 2D grid to be plotted with contour.
% 把 pdf 值 reshape 回 100x100，好畫 contour
Z1 = reshape(z1, gridSize, gridSize);
Z2 = reshape(z2, gridSize, gridSize);

% Plot the contour lines to show the pdf over the data.
% 畫兩個分佈的 pdf 等高線
[C, h] = contour(u, u, Z1);
[C, h] = contour(u, u, Z2);

axis([-6 6 -6 6])
title('Original Data and PDFs');

%set(h,'ShowText','on','TextStep',get(h,'LevelStep')*2);


%%====================================================
%% STEP 2: Choose initial values for the parameters.
% 設定 EM 演算法一開始的初始參數 (mu, sigma, phi)

% Set 'm' to the number of data points.
m = size(X, 1);   % 總共有多少筆資料

k = 2;  % The number of clusters. 要分成幾群
n = 2;  % The vector lengths. 每筆資料的維度

% Randomly select k data points to serve as the initial means.
% 隨便從資料裡抽兩點當作初始的 mean
indeces = randperm(m);
mu = X(indeces(1:k), :);

sigma = [];

% Use the overal covariance of the dataset as the initial variance for each cluster.
% 一開始先把整個 dataset 的 covariance 當作每一群的初始 sigma
for (j = 1 : k)
    sigma{j} = cov(X);
end

% Assign equal prior probabilities to each cluster.
% mixing coefficient 每群的權重 一開始全部平均 0.5 / 0.5，各占一半
phi = ones(1, k) * (1 / k);

%%===================================================
%% STEP 3: Run Expectation Maximization

% Matrix to hold the probability that each data point belongs to each cluster.
% One row per data point, one column per cluster.
% 開始跑 EM，不斷重複 E-step 和 M-step
W = zeros(m, k);  % W(i,j) = 第 i 筆資料屬於第 j 群的機率

% Loop until convergence.
% 最多跑 1000 次，如果收斂會提前跳出
for (iter = 1:1000)
    
    fprintf('  EM Iteration %d\n', iter);

    %%===============================================
    %% STEP 3a: Expectation
    %
    % Calculate the probability for each data point for each distribution.
    % 根據目前的 mu / sigma / phi，計算每筆資料屬於每群的機率
    
    % Matrix to hold the pdf value for each every data point for every cluster.
    % One row per data point, one column per cluster.
    pdf = zeros(m, k); % pdf(i,j) = 第 i 筆資料在第 j 個 Gaussian 下的機率密度
    
    % For each cluster...
    % 算第 j 群的 pdf
    for (j = 1 : k)       
        % Evaluate the Gaussian for all data points for cluster 'j'.
        pdf(:, j) = gaussianND(X, mu(j, :), sigma{j});
    end
    
    % Multiply each pdf value by the prior probability for cluster.
    %    pdf  [m  x  k]
    %    phi  [1  x  k]   
    %  pdf_w  [m  x  k]
    pdf_w = bsxfun(@times, pdf, phi); % pdf * phi，把 pdf 乘以群的先驗權重
    
    % Divide the weighted probabilities by the sum of weighted probabilities for each cluster.
    %   sum(pdf_w, 2) -- sum over the clusters.
    W = bsxfun(@rdivide, pdf_w, sum(pdf_w, 2)); % W = pdf_w / 各資料點加總
    
    %%===============================================
    %% STEP 3b: Maximization
    %%
    %% Calculate the probability for each data point for each distribution.
    % 依照剛算好的 W 來更新 mu / sigma / phi

    % Store the previous means.
    prevMu = mu;    % 記錄上一輪的 mean，用來檢查有沒有收斂
    
    % For each of the clusters...
    for (j = 1 : k)
    
        % Calculate the prior probability for cluster 'j'.
        phi(j) = mean(W(:, j), 1); % 更新 mixing coefficient
        % Calculate the new mean for cluster 'j' by taking the weighted
        % average of all data points.
        mu(j, :) = weightedAverage(W(:, j), X); % 更新 mean

        % Calculate the covariance matrix for cluster 'j' by taking the 
        % weighted average of the covariance for each training example. 
        
        sigma_k = zeros(n, n); % 更新共變異數
        
        % Subtract the cluster mean from all data points.
        Xm = bsxfun(@minus, X, mu(j, :)); % 每筆資料減掉該群 mean
        
        % Calculate the contribution of each training example to the covariance matrix.
        % 累計每筆資料的貢獻
        for (i = 1 : m)
            sigma_k = sigma_k + (W(i, j) .* (Xm(i, :)' * Xm(i, :)));
        end
        
        % Divide by the sum of weights.
        % 完整加權平均
        sigma{j} = sigma_k ./ sum(W(:, j));
    end
    
    % Check for convergence.
    % 如果 mu 完全沒變，收斂了，結束
    if (mu == prevMu)
        break
    end
            
% End of Expectation Maximization    
end

%%=====================================================
%% STEP 4: Plot the data points and their estimated pdfs.
% 把 EM 收斂後的分佈畫出來，看看有沒有成功抓到兩群的形狀

% Display a scatter plot of the two distributions.
figure(2);
hold off;
plot(X1(:, 1), X1(:, 2), 'bo'); % 原始第一群
hold on;
plot(X2(:, 1), X2(:, 2), 'ro'); % 原始第二群

set(gcf,'color','white') % White background for the figure.

plot(mu1(1), mu1(2), 'kx');  % 原本真實的 mean
plot(mu2(1), mu2(2), 'kx');

% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 100;
u = linspace(-6, 6, gridSize);
[A B] = meshgrid(u, u);
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
% 用 EM 估出的 mu、sigma 畫 pdf
z1 = gaussianND(gridX, mu(1, :), sigma{1});
z2 = gaussianND(gridX, mu(2, :), sigma{2});

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1 = reshape(z1, gridSize, gridSize);
Z2 = reshape(z2, gridSize, gridSize);

% Plot the contour lines to show the pdf over the data.
[C, h] = contour(u, u, Z1); % 估出來的第一群 pdf
[C, h] = contour(u, u, Z2); % 估出來的第二群 pdf
axis([-6 6 -6 6])

title('Original Data and Estimated PDFs'); % 原始資料 vs EM 估出來的結果