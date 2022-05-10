class ModelConfig:
    training_parameters = {
        'num_of_labels': 7,
        'epochs': 20,
        'model_name': 'cbert_model.pkl'
    }
    callback_parameters = {
        'time_stopping_seconds': 86400,
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 1e-4,
        'use_early_stopping': False,
        'use_time_stopping': True
    }
    optimizer_parameters = {
        'learning_rate': 1e-3,
        'weight_decay_rate': 1e-2,
        'weight_no_decay_rate': 0.0,
        'no_decay_weights': ['bias', 'gamma', 'beta']
    }
    hyperparameters = {
        'batch_size': 64,
        'max_sequence_length': 256,
        'bert_pretrained': 'bert-base-chinese',
        'bert_embedding_dim': 768,
        'bert_weight_norm_mean': 0.0,
        'bert_weight_norm_std': 2e-2,
    }
    augmentation_parameters = {
        'temperature_value': 5e-1,
        'sample_ratio': 2
    }
