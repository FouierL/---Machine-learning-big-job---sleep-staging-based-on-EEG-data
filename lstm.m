%% 导入数据
clc;clear;close all
%每个受试者的数据选取1000例训练模型，100例做测试集，100例做验证集
load ccq
data_train=features(1:1000,:);
data_test=features(1001:1100,:);
data_verify=features(1101:1200,:);
load gyf
data_train=cat(1,data_train,features(1:1000,:));
data_test=cat(1,data_test,features(1001:1100,:));
data_verify=cat(1,data_verify,features(1101:1200,:));
load pl
data_train=cat(1,data_train,features(1:1000,:));
data_test=cat(1,data_test,features(1001:1100,:));
data_verify=cat(1,data_verify,features(1101:1200,:));
load tyt
data_train=cat(1,data_train,features(1:1000,:));
data_test=cat(1,data_test,features(1001:1100,:));
data_verify=cat(1,data_verify,features(1101:1200,:));
load whr
data_train=cat(1,data_train,features(1:1000,:));
data_test=cat(1,data_test,features(1001:1100,:));
data_verify=cat(1,data_verify,features(1101:1200,:));
load wl
data_train=cat(1,data_train,features(1:1000,:));
data_test=cat(1,data_test,features(1001:1100,:));
data_verify=cat(1,data_verify,features(1101:1200,:));
load wm
data_train=cat(1,data_train,features(1:1000,:));
data_test=cat(1,data_test,features(1001:1100,:));
data_verify=cat(1,data_verify,features(1101:1200,:));
features_tr=zeros(7000,9);
features_v=zeros(700,9);
features_te=zeros(700,9);
%划分训练集，测试集，验证集与标签
label_train=data_train(:,7681);
label_test=data_test(:,7681);
label_verify=data_verify(:,7681);
data_train=data_train(:,1:7680);
data_test=data_test(:,1:7680);
data_verify=data_verify(:,1:7680);
for i=1:700
    if label_verify(i,1)==2
        label_verify(i,1)=1;
    end
    if label_test(i,1)==2
        label_test(i,1)=1;
    end
end
for i=1:7000
    if label_train(i,1)==2
        label_train(i,1)=1;
    end
end
%% 特征提取
delta=[0.5 3];sita=[4 7];alpha=[8 13];beta=[14 30];
for k=1:size(data_train,1)
    features_tr(k,1)=max(data_train(k,:));%最大值
    features_tr(k,2)=min(data_train(k,:));%最小值
    features_tr(k,3)=median(data_train(k,:));%中位数
    features_tr(k,4)=mean(data_train(k,:));%均值
    features_tr(k,5)=std(data_train(k,:));%标准差
    features_tr(k,6)=var(data_train(k,:));%方差
    features_tr(k,7)=sqrt(sum(data_train(k,:).^2));% 均方根
    features_tr(k,8)=mean(abs(data_train(k,:)-mean(data_train(k,:))));%均值差
    features_tr(k,9)=skewness(data_train(k,:));%偏度
    features_tr(k,10)=kurtosis(data_train(k,:));%峰度
    features_tr(k,11:15)=Features_f(data_train(k,:));%频域特征
end
for k=1:size(data_test,1)
     features_te(k,1)=max(data_test(k,:));%最大值
    features_te(k,2)=min(data_test(k,:));%最小值
    features_te(k,3)=median(data_test(k,:));%中位数
    features_te(k,4)=mean(data_test(k,:));%均值
    features_te(k,5)=std(data_test(k,:));%标准差
    features_te(k,6)=var(data_test(k,:));%方差
    features_te(k,7)=sqrt(sum(data_test(k,:).^2));% 均方根
    features_te(k,8)=mean(abs(data_test(k,:)-mean(data_test(k,:))));%均值差
    features_te(k,9)=skewness(data_test(k,:));%偏度
    features_te(k,10)=kurtosis(data_test(k,:));%峰度
    features_te(k,11:15)=Features_f(data_test(k,:));%频域特征
end
for k=1:size(data_verify,1)
    features_v(k,1)=max(data_verify(k,:));%最大值
    features_v(k,2)=min(data_verify(k,:));%最小值
    features_v(k,3)=median(data_verify(k,:));%中位数
    features_v(k,4)=mean(data_verify(k,:));%均值
    features_v(k,5)=std(data_verify(k,:));%标准差
    features_v(k,6)=var(data_verify(k,:));%方差
    features_v(k,7)=sqrt(sum(data_verify(k,:).^2));% 均方根
    features_v(k,8)=mean(abs(data_verify(k,:)-mean(data_verify(k,:))));%均值差
    features_v(k,9)=skewness(data_verify(k,:));%偏度
    features_v(k,10)=kurtosis(data_verify(k,:));%峰度
    features_v(k,11:15)=Features_f(data_verify(k,:));%频域特征
end
%% 创建变量
% features_tr=mts.train;
% label_train=categorical(mts.trainlabels);%将数值数组转化为类别数组
% features_te=mts.test;
% label_test=categorical(mts.testlabels);
label_train=label_train';
label_test=label_test';
features_tr=features_tr';
features_te=features_te';
label_train = categorical(label_train);
label_test = categorical(label_test);
features_tr = num2cell(features_tr,1);
features_te = num2cell(features_te,1);
%% 构建LSTM网络
inputSize = 15;%特征的维度
numHiddenUnits = 500;%LSTM网路包含的隐藏单元数目
numClasses = 4;%label标签的种数,该例子中为人数

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

maxEpochs = 100;%最大训练周期数
miniBatchSize = 27;%分块尺寸

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');
%% 训练
net = trainNetwork(features_tr',label_train',layers, options);
%% 预测
YPred = classify(net,features_te, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');
%% 精确度检验
acc = sum(YPred == label_test)./numel(label_test)
