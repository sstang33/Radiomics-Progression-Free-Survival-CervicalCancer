clear all,close all,clc

filepath = 'D:\Cervix Cancer\code\survival prediction 5CV DiseaseFree\MR\NotCpltExc NeverDisFreeModified\';
filename = 'Rad_PFS_maxFea6_Average.xlsx';

[~,~,raw] = xlsread([filepath,filename],'Combine');
data = cell2mat(raw(2:end,2:end));
mrn = raw(2:end,1);
title = raw(1,:);
duration = data(:,2);
dur_mon = duration/365*12;
relapse = data(:,3);

%% 12 mon RFS
clear idx0 idx1 idx2 idx3 idx4 idx
data_12 = data;
idx3 = find(dur_mon>12);
idx4 = find(relapse==1);
idx0 = intersect(idx3,idx4);
data_12(idx0,3) = 0;

idx1 = find(dur_mon<12);
idx2 = find(relapse==0);
idx = intersect(idx1,idx2);
data_12(idx,:) = [];
mrn_12 = mrn;
mrn_12(idx) = [];

xlswrite([filepath,filename],title,'12monRFS','A1');
xlswrite([filepath,filename],mrn_12,'12monRFS','A2');
xlswrite([filepath,filename],data_12,'12monRFS','B2');
disp(['total: ',num2str(size(data_12,1)),'; relapse:',num2str(sum(data_12(:,3))),'; NonRelapse: ',num2str(size(data_12,1)-sum(data_12(:,3)))])

%% 24 mon RFS
clear idx0 idx1 idx2 idx3 idx1
data_24 = data;
idx3 = find(dur_mon>24);
idx4 = find(relapse==1);
idx0 = intersect(idx3,idx4);
data_24(idx0,3) = 0;

idx1 = find(dur_mon<24);
idx2 = find(relapse==0);
idx = intersect(idx1,idx2);
data_24(idx,:) = [];
mrn_24 = mrn;
mrn_24(idx) = [];

xlswrite([filepath,filename],title,'24monRFS','A1');
xlswrite([filepath,filename],mrn_24,'24monRFS','A2');
xlswrite([filepath,filename],data_24,'24monRFS','B2');
disp(['total: ',num2str(size(data_24,1)),'; relapse:',num2str(sum(data_24(:,3))),'; NonRelapse: ',num2str(size(data_24,1)-sum(data_24(:,3)))])

%% 36 mon RFS
clear idx0 idx1 idx2 idx3 idx1
data_36 = data;
idx3 = find(dur_mon>36);
idx4 = find(relapse==1);
idx0 = intersect(idx3,idx4);
data_36(idx0,3) = 0;

idx1 = find(dur_mon<36);
idx2 = find(relapse==0);
idx = intersect(idx1,idx2);
data_36(idx,:) = [];
mrn_36 = mrn;
mrn_36(idx) = [];

xlswrite([filepath,filename],title,'36monRFS','A1');
xlswrite([filepath,filename],mrn_36,'36monRFS','A2');
xlswrite([filepath,filename],data_36,'36monRFS','B2');

disp(['total: ',num2str(size(data_36,1)),'; relapse:',num2str(sum(data_36(:,3))),'; NonRelapse: ',num2str(size(data_36,1)-sum(data_36(:,3)))])