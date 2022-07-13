function res = PQprtMOV (MOV, ODG)
% Print MOV values (PEAQ Basic version)

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:47 $

N = length (MOV);
PQ_NMOV_B = 11;
PQ_NMOV_A = 5;

% Calculate 2f-model value
MMS = twofmodel(MOV(7),MOV(5));

fprintf ('Model Output Variables:\n');
if (N == PQ_NMOV_B)
    fprintf ('   BandwidthRefB: %g\n', MOV(1));
    fprintf ('  BandwidthTestB: %g\n', MOV(2));
    fprintf ('      Total NMRB: %g\n', MOV(3));
    fprintf ('    WinModDiff1B: %g\n', MOV(4));
    fprintf ('            ADBB: %g\n', MOV(5));
    fprintf ('            EHSB: %g\n', MOV(6));
    fprintf ('    AvgModDiff1B: %g\n', MOV(7));
    fprintf ('    AvgModDiff2B: %g\n', MOV(8));
    fprintf ('   RmsNoiseLoudB: %g\n', MOV(9));
    fprintf ('           MFPDB: %g\n', MOV(10));
    fprintf ('  RelDistFramesB: %g\n', MOV(11));
    res.BandwidthRefB = MOV(1);
    res.BandwidthTestB = MOV(2);
    res.TotalNMRB = MOV(3);
    res.WinModDiff1B = MOV(4);
    res.ADBB = MOV(5);
    res.EHSB = MOV(6);
    res.AvgModDiff1B = MOV(7);
    res.AvgModDiff2B = MOV(8);
    res.RmsNoiseLoudB = MOV(9);
    res.MFPDB = MOV(10);
    res.RelDistFramesB = MOV(11);
elseif (N == NMOV_A)
    fprintf ('        RmsModDiffA: %g\n', MOV(1));
    fprintf ('  RmsNoiseLoudAsymA: %g\n', MOV(2));
    fprintf ('     Segmental NMRB: %g\n', MOV(3));
    fprintf ('               EHSB: %g\n', MOV(4));
    fprintf ('        AvgLinDistA: %g\n', MOV(5));
    res.RmsModDiffA = MOV(1);
    res.RmsNoiseLoudAsymA = MOV(2);
    res.SegmentalNMRB = MOV(3);
    res.EHSB = MOV(4);
    res.AvgLinDistA = MOV(5);
else
    error ('Invalid number of MOVs');
end

fprintf ('Objective Difference Grade: %.3f\n', ODG);
fprintf ('2f-model: %.4f\n', MMS);
res.ODG = ODG;
res.TFM = MMS;

