from easydict import EasyDict as edict  # Import EasyDict for convenient configuration management

# Create an EasyDict object to store configuration settings
config = edict()

# Validation targets for face recognition
config.val_targets = ['lfw', 'cfp_fp', "agedb_30", 'calfw', 'cplfw']

# Path to the training dataset
config.rec = "<your path>/data/faces_emore/" 
config.num_classes = 85742  # Number of classes in the training dataset
config.num_image = 5822653  # Number of images in the training dataset

# Paths for age and gender datasets
config.age_gender_data_path = "<your path>/data/AIO_train"
config.age_gender_data_list = ["IMDB", "WIKI", "Adience", "MORPH"]  # List of datasets for age and gender

# Paths for CelebA dataset
config.CelebA_train_data = "<your path>/data/CelebA/data"
config.CelebA_train_label = "<your path>/data/AIO_train/CelebA/label.txt"
config.CelebA_val_data = "<your path>/data/CelebA/data"
config.CelebA_val_label = "<your path>/data/AIO_val/CelebA/label.txt"
config.CelebA_test_data = "<your path>/data/CelebA/data"
config.CelebA_test_label = "<your path>/data/AIO_test/CelebA/label.txt"

# Paths for FGnet dataset
config.FGnet_data = "<your path>/data/AIO_val/FGnet/data"
config.FGnet_label = "<your path>/data/AIO_val/FGnet/label.txt"

# Paths for RAF dataset
config.RAF_data = "<your path>/data/RAF"
config.RAF_label = "<your path>/data/RAF_/basic/list_patition_label.txt"

# Paths for AffectNet dataset
config.AffectNet_data = "<your path>/data/AffectNet/data"
config.AffectNet_label = "<your path>/data/AffectNet/label.txt"

# Paths for LAP dataset
config.LAP_train_data = "<your path>/data/AIO_test/LAP_finetuning/data"
config.LAP_train_label = "<your path>/data/AIO_test/LAP_finetuning/label.csv"
config.LAP_test_data = "<your path>/data/AIO_test/LAP_test/data"
config.LAP_test_label = "<your path>/data/AIO_test/LAP_test/label.csv"

# Paths for CLAP dataset
config.CLAP_train_data = "<your path>/data/AIO_test/LAP_finetuning/data"
config.CLAP_train_label = "<your path>/data/AIO_test/LAP_finetuning/label.csv"
config.CLAP_val_data = "<your path>/data/AIO_test/LAP_test/data"
config.CLAP_val_label = "<your path>/data/AIO_test/LAP_test/label.csv"
config.CLAP_test_data = "<your path>/data/LAP_test/test"
config.CLAP_test_label = "<your path>/data/LAP_test/test.csv"

# Image size and batch sizes for different datasets
config.img_size = 112  # Image size for input data
config.batch_size = 128  # General batch size for training
config.recognition_bz = 32  # Batch size for face recognition
config.age_gender_bz = 32  # Batch size for age and gender training
config.CelebA_bz = 32  # Batch size for CelebA dataset
config.expression_bz = 32  # Batch size for expression training

# Training parameters
config.train_num_workers = 2  # Number of workers for training data loader
config.train_pin_memory = True  # Whether to pin memory for training data loader

# Validation parameters
config.val_batch_size = 128  # Batch size for validation
config.val_num_workers = 0  # Number of workers for validation data loader
config.val_pin_memory = True  # Whether to pin memory for validation data loader

# Image interpolation method
config.INTERPOLATION = 'bicubic'

# RAF dataset parameters
config.RAF_NUM_CLASSES = 7  # Number of classes in RAF dataset
config.RAF_LABEL_SMOOTHING = 0.1  # Label smoothing factor for RAF dataset

# Augmentation parameters
config.AUG_COLOR_JITTER = 0.4  # Color jitter factor
config.AUG_AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'  # Auto augmentation policy
config.AUG_REPROB = 0.25  # Random erasing probability
config.AUG_REMODE = 'pixel'  # Random erasing mode
config.AUG_RECOUNT = 1  # Number of random erasing attempts
config.AUG_MIXUP = 0.0  # Mixup alpha value
config.AUG_CUTMIX = 0.0  # Cutmix alpha value
config.AUG_CUTMIX_MINMAX = None  # Cutmix min/max values
config.AUG_MIXUP_PROB = 1.0  # Mixup probability
config.AUG_MIXUP_SWITCH_PROB = 0.5  # Mixup switch probability
config.AUG_MIXUP_MODE = 'batch'  # Mixup mode

# Scale augmentation parameters
config.AUG_SCALE_SET = True  # Whether to use scale augmentation
config.AUG_SCALE_SCALE = (1.0, 1.0)  # Scale range
config.AUG_SCALE_RATIO = (1.0, 1.0)  # Scale ratio range

# Network architecture
config.network = "swin_t"  # Network architecture

# Feature Attention Module (FAM) parameters
config.fam_kernel_size = 3  # Kernel size for FAM
config.fam_in_chans = 2112  # Number of input channels for FAM
config.fam_conv_shared = False  # Whether to share convolution layers in FAM
config.fam_conv_mode = "split"  # Convolution mode in FAM
config.fam_channel_attention = "CBAM"  # Channel attention mechanism in FAM
config.fam_spatial_attention = None  # Spatial attention mechanism in FAM
config.fam_pooling = "max"  # Pooling method in FAM
config.fam_la_num_list = [2 for j in range(11)]  # List of numbers for FAM
config.fam_feature = "all"  # Feature type for FAM
config.fam = "3x3_2112_F_s_C_N_max"  # FAM configuration string

# Embedding size
config.embedding_size = 512  # Size of the embedding vector

# Training resume parameters
config.resume = False  # Whether to resume training
config.resume_step = 0  # Step to resume from
config.init = True  # Whether to initialize the model
config.init_model = "<your path>/insightface/results/arcface_torch/init/"  # Path to the initialization model

# Learning rate schedule parameters
config.warmup_step = 8000  # Number of warmup steps
config.total_step = 80000  # Total number of training steps
config.optimizer = "adamw"  # Optimizer type
config.lr = 5e-4  # Learning rate
config.weight_decay = 0.05  # Weight decay factor
config.lr_name = 'cosine'  # Learning rate schedule name
config.warmup_lr = 5e-7  # Warmup learning rate
config.min_lr = 5e-6  # Minimum learning rate
config.decay_epoch = 10  # Epoch for learning rate decay
config.decay_rate = 0.1  # Decay rate for learning rate

# Loss parameters
config.margin_list = (1.0, 0.0, 0.4)  # Margin values for loss function
config.sample_rate = 0.3  # Sampling rate for loss function
config.interclass_filtering_threshold = 0  # Interclass filtering threshold

# Loss weights
config.recognition_loss_weight = 1.0  # Weight for recognition loss
config.analysis_loss_weights = [1.0 for j in range(42)]  # Weights for analysis loss

# Mixed precision training
config.fp16 = True  # Whether to use mixed precision training
config.dali = False  # Whether to use DALI for data loading
config.seed = 2048  # Random seed for reproducibility

# Save and result parameters
config.save_all_states = True  # Whether to save all model states
config.result = "<your path>/results"  # Path to save results

# Logging parameters
config.verbose = 2000  # Frequency of logging messages
config.save_verbose = 4000  # Frequency of saving model states
config.frequent = 10  # Frequency of logging during training