function MMS = twofmodel (AvgModDiff1,ADB)
% Calculates parameter of 2fmodel, based on:
% https://www.audiolabs-erlangen.de/resources/2019-WASPAA-SEBASS/
% INPUTS:
% - AvgModDiff1: output MOV from PQevalAudio
% - ADB: output MOV from PQevalAudio
% OUTPUT:
% - MMS: output of 2f-model (score in range from 0 to 100).
MMS = 56.1345 / (1 + (-0.0282 * AvgModDiff1 - 0.8628)^2) - 27.1451 * ADB + 86.3515;