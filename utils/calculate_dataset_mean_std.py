from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob


class CocoFlatImageDataset(Dataset):
    def __init__(self, img_folder_path, transform=None):
        """
        Args:
            img_folder_path (str): 包含所有图像的文件夹路径 (例如 'D:/Python/datasets/coco2014/train2014')。
            transform (callable, optional): 应用于图像的变换。
        """
        self.img_folder_path = img_folder_path
        self.transform = transform

        self.image_paths = []
        for ext in ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]:
            self.image_paths.extend(
                glob.glob(os.path.join(img_folder_path, f"*.{ext}"))
            )

        if not self.image_paths:
            raise FileNotFoundError(
                f"No image files found in {img_folder_path}. Please check the path and image extensions."
            )

        print(f"Found {len(self.image_paths)} images in {img_folder_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0


def calculate_dataset_mean_std(
    dataset_root, resize_dim=(416, 416), batch_size=64, num_workers=4
):
    """
    计算给定图像数据集的每个通道的均值和标准差。

    Args:
        dataset_root (str): 数据集根目录的路径，期望是ImageFolder兼容的结构。
        resize_dim (tuple): (height, width) 图片将被缩放到的尺寸，用于统一大小。
        batch_size (int): DataLoader的批次大小。
        num_workers (int): DataLoader的并行工作进程数。

    Returns:
        tuple: (mean_list, std_list)，包含每个通道的均值和标准差的列表。
    """
    if not os.path.exists(dataset_root):
        raise ValueError(f"{dataset_root} does not exist.")

    transform_for_stats = transforms.Compose(
        [transforms.Resize(resize_dim), transforms.ToTensor()]
    )

    dataset = CocoFlatImageDataset(
        img_folder_path=dataset_root, transform=transform_for_stats
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    mean = 0.0
    std = 0.0
    total_images_count = 0

    print(f"开始计算数据集 '{dataset_root}' 的均值和标准差...")
    for images, _ in tqdm(data_loader):
        batch_count = images.size(0)
        # 将图像展平为 (batch_size, channels, height*width)
        images = images.view(batch_count, images.size(1), -1)

        # 累加每个通道的均值
        mean += images.mean(2).sum(0)
        # 累加每个通道的标准差
        std += images.std(2).sum(0)

        total_images_count += batch_count

    final_mean = mean / total_images_count
    final_std = std / total_images_count

    print("\n计算完成！")
    print(f"数据集的均值 (mean): {final_mean}")
    print(f"数据集的标准差 (std): {final_std}")

    return final_mean, final_std


if __name__ == "__main__":
    my_dataset_path = r"D:\Python\datasets\coco2014\train2014"

    dataset_mean, dataset_std = calculate_dataset_mean_std(
        dataset_root=my_dataset_path,
        resize_dim=(416, 416),
        batch_size=128,
        num_workers=4,
    )

    print(f"\n可以在 transforms.Normalize 中使用的均值: {dataset_mean}")
    print(f"可以在 transforms.Normalize 中使用的标准差: {dataset_std}")
