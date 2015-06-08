PARAM.Classifier_Type = 'svm-struct';
PARAM.Train_Pos_IoU_Threshold = 0.7;
PARAM.SVM_Epoch = 1;
PARAM.SVM_C = 1;
PARAM.SVM_bias = 10;
PARAM.SVM_PosW = 2;
PARAM.SVM_ConstType = 'maxmax';
PARAM.SVM_HingeLossType = 'L1';

SPECIFIC_DIRS.Train            = 'Train/svm_struct';
SPECIFIC_DIRS.Test4GPTrain     = 'Test4GPTrain/svm_struct';
SPECIFIC_DIRS.GPTrain          = 'GPTrain/svm_struct';

