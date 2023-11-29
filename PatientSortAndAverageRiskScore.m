clear all,close all,clc

filepath = 'D:\Cervix Cancer\code\survival prediction 5CV DiseaseFree\MR\NotCpltExc NeverDisFreeModified\';
filename = 'Rad_PFS_maxFea6_';
writefile = [filename,'Average.xlsx'];
pat_num = 105;

for i = 0:4
    RiskFile = [filepath,filename,num2str(i),'.xlsx'];
    [~,~,raw] = xlsread(RiskFile,'Combine');
    mrn = string(raw(2:end,1));
    ids = string(regexp(mrn,'\d+','match'));
    num = str2double(ids);
    data = cell2mat(raw(2:pat_num+1,2:end));
    data_all = [num,data];
    sortData(:,:,i+1) = sortrows(data_all,1);
end

data_temp = mean(sortData,3);

%%
num_temp = data_temp(:,1);
clear newname
for ip = 1:length(num_temp)
    newname = cellstr(['CRV_',num2str(num_temp(ip), '%03d')]);
    xlRange = ['A',num2str(ip+1)];
    xlswrite([filepath,writefile],newname,'Combine',xlRange)
end

title = raw(1,:);
xlswrite([filepath,writefile],title,'Combine','A1') 
xlswrite([filepath,writefile],data_temp(:,2:end),'Combine','B2') 
