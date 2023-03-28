%% 导入数据
clc;clear;close all
%每个受试者的数据选取1000例训练模型，100例做测试集，100例做验证集
load ccq
rowrank = randperm(size(features, 1)); % size获得a的行数，randperm打乱各行的顺序
features = features(rowrank,:);              % 按照rowrank重新排列各行，注意rowrank的位置
data_train=features(1:1000,:);
data_test=features(1001:1100,:);
data_verify=features(1101:1200,:);
load gyf
rowrank = randperm(size(features, 1));
features = features(rowrank,:);
data_train=cat(1,data_train,features(1:1000,:));
data_test=cat(1,data_test,features(1001:1100,:));
data_verify=cat(1,data_verify,features(1101:1200,:));
load pl
rowrank = randperm(size(features, 1));
features = features(rowrank,:);
data_train=cat(1,data_train,features(1:1000,:));
data_test=cat(1,data_test,features(1001:1100,:));
data_verify=cat(1,data_verify,features(1101:1200,:));
load tyt
rowrank = randperm(size(features, 1));
features = features(rowrank,:);
data_train=cat(1,data_train,features(1:1000,:));
data_test=cat(1,data_test,features(1001:1100,:));
data_verify=cat(1,data_verify,features(1101:1200,:));
load whr
rowrank = randperm(size(features, 1));
features = features(rowrank,:);
data_train=cat(1,data_train,features(1:1000,:));
data_test=cat(1,data_test,features(1001:1100,:));
data_verify=cat(1,data_verify,features(1101:1200,:));
load wl
rowrank = randperm(size(features, 1));
features = features(rowrank,:);
data_train=cat(1,data_train,features(1:1000,:));
data_test=cat(1,data_test,features(1001:1100,:));
data_verify=cat(1,data_verify,features(1101:1200,:));
load wm
rowrank = randperm(size(features, 1));
features = features(rowrank,:);
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
%% 
