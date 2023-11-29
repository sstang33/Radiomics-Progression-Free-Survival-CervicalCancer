clear all,close all,clc

filepath = 'D:\Cervix Cancer\code\survival prediction 5CV DiseaseFree\MR\NotCpltExc NeverDisFreeModified\';
filename = 'Rad_PFS_maxFea7_';
pat_num = 105;

for file = 5:14
    data = [];
    for i = 0:4
        sheet = ['CT',num2str(i)];
        fileread = [filepath,filename,num2str(file),'.xlsx'];
        [~,~,temp] = xlsread(fileread,sheet);
        data = [data;temp(2:end,2:end)];
    end
    title = {'PatientsID','Prediction','Survival','Event'};
    xlswrite(fileread,title,'Combine','A1') 
    xlswrite(fileread,data,'Combine','A2') 
end
