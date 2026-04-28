# config.py — default hyperparameters (overridden by argparse in main.py)

DEVICE     = "cuda"
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-4
LORA_RANK  = 4
PATIENCE   = 3
DROPOUT    = 0.0
METHOD     = "cosine"   # "cosine" or "classifier"
