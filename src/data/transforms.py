from torchvision.transforms import v2

def get_train_transforms(
        img_size:int=224,
        mean: list[float] | None = None,
        std: list[float] | None = None
    ) -> v2.Compose:

    """
        augmented transforms used during training
    """

    # mean and standard deviation from the ImageNet dataset
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    pad_size = img_size + 32

    return v2.Compose(
        [
            v2.Resize((pad_size, pad_size)),
            v2.RandomResizedCrop(size=(img_size, img_size)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.RandomGrayscale(p=0.1),
            v2.RandomRotation(degrees=[-10, 10]),
            v2.ToTensor(),
            v2.Normalize(mean=mean, std=std),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.3, 0.3), value=0)  # erasing all affected pixels
        ]
    )

def get_deterministic_transforms(
        img_size:int=200,
        mean: list[float] | None = None,
        std: list[float] | None = None
) -> v2.Compose:

    """
        human-readable transforms for validation / testing
    """

    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    return v2.Compose(
        [
            v2.Resize(img_size, img_size),
            v2.ToTensor(),
            v2.Normalize(mean=mean, std=std)
        ]
    )