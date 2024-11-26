from .imports import *

data_path = Path('data/')
image_path = data_path / 'virus'
train_dir = image_path / 'train'
test_dir = image_path / 'test'

def dataloader(BATCH_SIZE: int = 32, num_workers: int = 1, pin_memory: bool = False):

    custom_transforms = torchvision.transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=custom_transforms,
                                      target_transform=None)

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=custom_transforms,
                                     target_transform=None)

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    return train_dataloader, test_dataloader