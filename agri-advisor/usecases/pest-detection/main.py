from utils.data_loaders import DataModule
from config import Config
from utils.trainer_utils import EfficientNetB3Trainer
from torchvision import transforms

if __name__ == "__main__":

    train_txt_path = Config.ROOT + "\\train.txt"
    val_txt_path = Config.ROOT + "\\val.txt"
    test_txt_path = Config.ROOT + "\\test.txt"

    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch_size = Config.BATCH_SIZE

    data_module = DataModule(
        train_txt_path=train_txt_path,
        val_txt_path=val_txt_path,
        test_txt_path=test_txt_path,
        train_transform=train_transform,
        val_test_transform=val_test_transform,
        batch_size=batch_size
    )

    train_loader, val_loader, test_loader = data_module.get_loaders()

    print(f"Train loader batches: {len(train_loader)}")
    print(f"Validation loader batches: {len(val_loader)}")
    print(f"Test loader batches: {len(test_loader)}")

    trainer = EfficientNetB3Trainer()
    test_acc, test_classwise = trainer.fit(train_loader, val_loader, test_loader)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Classwise Test Accuracy (first 10): {test_classwise[:10]}")
