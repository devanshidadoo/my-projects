import os, re

def get_next_ckpt():
        d, b = "checkpoints", "efficientnet_b3_best.pth"
        os.makedirs(d, exist_ok=True)
        vs = [int(m.group(1)) for f in os.listdir(d) if (m:=re.search(r"_v(\d+)$", f))]
        return f"{d}/{b}_v{max(vs,default=0)+1}"

class Config:
    
    #TRAINING
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_CLASSES = 102
    EPOCHS = 10
    DEVICE = "cpu"
    MISSING_FILES_LOG = "missing_files.txt"
    ROOT = r'C:\Users\subar\OneDrive\Desktop\capitalone\usecases\pest-detection\ip102_v1.1\ip102_v1.1'
    RUN_CHECKPOINT = "checkpoints\efficientnet_ip102.pth"
    FREEZE_FEATURES = True
    LR = 1e-3
    
    SAVE_BEST = get_next_ckpt()
    WEIGHT_DECAY = 1e-4
    
    #INFERENCE
    INFERENCE_MODEL = "checkpoints\efficientnet_ip102.pth"
    
 
