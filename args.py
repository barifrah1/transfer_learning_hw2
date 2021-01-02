class NetArgs:
    lr = 1e-4
    lr_finetune = 1e-5
    weight_decay = 5e-6
    num_epochs = 5000
    early_stopping_min_improvement = 1e-7
    early_stopping_num_epochs = 5
    batch_size = 150
    train_size = 2*800
    test_size = 2*250
    X_train_extracted_from_resnet18_file_name = "X_extracted_features_train18.pt"
    y_train_extracted_from_resnet18_file_name = "y_extracted_features_train18.pt"
    X_test_extracted_from_resnet18_file_name = "X_extracted_features_test18.pt"
    y_test_extracted_from_resnet18_file_name = "y_extracted_features_test18.pt"
    X_train_extracted_from_resnet34_file_name = "X_extracted_features_train34.pt"
    y_train_extracted_from_resnet34_file_name = "y_extracted_features_train34.pt"
    X_test_extracted_from_resnet34_file_name = "X_extracted_features_test34.pt"
    y_test_extracted_from_resnet34_file_name = "y_extracted_features_test34.pt"
