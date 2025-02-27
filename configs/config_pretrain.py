from easydict import EasyDict as edict  # Import EasyDict for convenient configuration management

# Create an EasyDict object to store configuration settings
config = edict()

# Validation targets for face recognition
config.val_targets = ['lfw', 'cfp_fp', "agedb_30", 'calfw', 'cplfw']  # List of validation datasets

# Path to the training dataset
config.rec = "<your path>/data/faces_emore/"  # Path to the training dataset

# Training dataset parameters
config.num_classes = 85742  # Number of classes in the training dataset
config.num_image = 5822653  # Number of images in the training dataset

# Training parameters
config.batch_size = 128  # Batch size for training
config.num_workers = 8  # Number of workers for data loading
config.network = "swin_t"  # Network architecture to use

# Embedding and resume parameters
config.embedding_size = 512  # Size of the embedding vector
config.resume = False  # Whether to resume training from a checkpoint

# Epoch parameters
config.warmup_epoch = 5  # Number of warmup epochs
config.num_epoch = 40  # Total number of training epochs

# Optimizer parameters
config.optimizer = "adamw"  # Optimizer type
config.lr = 5e-4  # Learning rate
config.weight_decay = 0.05  # Weight decay factor

# Learning rate schedule parameters
config.lr_name = 'cosine'  # Learning rate schedule name
config.warmup_lr = 5e-7  # Warmup learning rate
config.min_lr = 5e-6  # Minimum learning rate
config.decay_epoch = 10  # Epoch for learning rate decay
config.decay_rate = 0.1  # Decay rate for learning rate

# Loss parameters
config.margin_list = (1.0, 0.0, 0.4)  # Margin values for loss function
config.sample_rate = 1.0  # Sampling rate for loss function
config.interclass_filtering_threshold = 0  # Interclass filtering threshold

# Mixed precision and other training parameters
config.fp16 = True  # Whether to use mixed precision training
config.dali = False  # Whether to use DALI for data loading
config.seed = 2048  # Random seed for reproducibility
config.save_all_states = True  # Whether to save all model states
config.result = "<your path>/results"  # Path to save results

# Logging parameters
config.verbose = 2000  # Frequency of logging messages
config.frequent = 10  # Frequency of logging during training