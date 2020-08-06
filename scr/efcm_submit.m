clear;
clc;
close all;
rng('default')
 
lamda = 2;
max_iter = 1000;        % 最大ループ回数
min_impro = 1e-6;       % 収束の閾値
n_cluster = 2          % 初期クラスタ数

% データ読み込み
dataname = 'seeds';
load seeds
data = SeedsInputs;
label = SeedsTargets;
 
[n_data, d_data]= size(data);
 
% メンバシップ値の初期化
randam_U = rand(n_cluster, n_data); %[0,1]のランダムな値を生成
col_sum = sum(randam_U,1);
U = randam_U./col_sum(ones(n_cluster, 1), :);%データ点ごとにメンバシップの和が1になるように正規化
 
objective_function = zeros(max_iter, 1);
 
% Main loop
for i = 1:max_iter
 
    center = U*data./((ones(size(data, 2), 1)*sum(U,2)')'); %クラスタ中心決定
    dist = pdist2(center, data);                            %クラスタ中心から各データ点への距離計算
    upper = exp(-lamda^(-1)*(dist.^2));                     %メンバシップ値の分子   
    U_new = upper./(ones(n_cluster, 1)*sum(upper));         %メンバシップ値計算   
    
    objective_function(i) = sum(sum((dist.^2).*U))+ lamda*sum(sum(log(U).*U));  % 目的関数の値を計算
    % 目的関数の値が収束しているか確認
    if i>1
        if abs(objective_function(i) - objective_function(i-1)) < min_impro
            break;
        end
    end  
    U = U_new;  %メンバシップ値更新
end
 
%PC
PC_C = (U).^2;
PC = 1/n_data*sum(sum(PC_C,2),1);
 
%PE
PE_C = -U.*log(U);
PE = 1/n_data*sum(sum(PE_C,2),1);
 
%Xie and Beni's index
Compactness = sum(sum((dist.^2).*U))+ lamda*sum(sum(log(U).*U)); 
dist_center = pdist2(center,center);
Separateness = n_data*min(dist_center(~(dist_center==tril(dist_center))));
XB = Compactness/Separateness;
 
Result.PC = PC;
Result.PE = PE;
Result.XB = XB;
%% save data
% save U
variable=strcat('U');
filename=strcat('U_',dataname,'_',num2str(n_cluster));
save(filename,variable);
% save center
variable=strcat('center');
filename=strcat('center_',dataname,'_',num2str(n_cluster));
save(filename,variable);
% save center
variable=strcat('Result');
filename=strcat('Result_',dataname,'_',num2str(n_cluster));
save(filename,variable);
