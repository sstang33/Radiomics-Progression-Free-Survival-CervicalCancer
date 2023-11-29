clear all,close all,clc

filepath = 'D:\Cervix Cancer\code\survival prediction 5CV DiseaseFree\Combined\NotCpltExc NeverDisFreeModified\';
MRname = 'Validation_Prediction_Expectation_OriFeature_HRselFea_OS_UpdateLastFU_NotCpltExc_NeverDisFreeModified_Average.xlsx';
Cliname = 'CliFea_CervixCancer_2Features_OS_Death_NotCpltExc_NeverDisFreeModi_Average.xlsx';
writefile = 'CombineMRCli_OS_Death_HRselFea_NotCpltExc_NeverDisFreeModi_PredictionAverage.xlsx';
pat_num = 105;

[~,~,rawMR] = xlsread([filepath,MRname],'Combine');
[~,~,rawCli] = xlsread([filepath,Cliname],'Combine');

% check patient ID and title
ptID_MR = cell2mat(rawMR(2:pat_num+1,1));
ptID_Cli = cell2mat(rawCli(2:pat_num+1,1));
title_MR = rawMR(1,:);
title_Cli = rawCli(1,:);
if ~isequal(ptID_MR,ptID_Cli) || ~isequal(title_MR,title_Cli)
    msg = 'Patient ID or title does not match';
    error(msg)
end

%% weighted average prediction
patID = rawMR(2:pat_num+1,1);
duration = cell2mat(rawMR(2:pat_num+1,3));
recurrence = cell2mat(rawMR(2:pat_num+1,4));
predMR = cell2mat(rawMR(2:pat_num+1,2));
predCli = cell2mat(rawCli(2:pat_num+1,2));

wivalue = 0.5;
predAve = 0.5*predCli+0.5*predMR;
data = [predAve,duration,recurrence];

% write to excel
sheet = 'Combine_MRCli_Ave';
xlswrite([filepath,writefile],title_MR,sheet,'A1') 
xlswrite([filepath,writefile],patID,sheet,'A2') 
xlswrite([filepath,writefile],data,sheet,'B2') 



