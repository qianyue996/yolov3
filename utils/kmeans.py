import numpy as np
from tqdm import tqdm


def main():
    boxes = []
    with open("coco_train.txt", "r", encoding="utf-8") as f:
        for line in f.readlines():
            for box in line.strip("\n").split(" ")[1:]:
                x1, y1, x2, y2, id = map(float, box.split(","))
                w, h = x2 - x1, y2 - y1
                boxes.append((w, h))
    kmeans(boxes)


def kmeans(boxes: list, k=9):
    boxes = np.array(boxes)
    n = boxes.shape[0]
    clusters = boxes[np.random.choice(n, k, replace=False)]

    last_clusters = np.zeros((n,))
    while True:
        distances = []
        for box in tqdm(boxes):
            distance = 1 - iou(box, clusters)
            distances.append(distance)
        distances = np.array(distances)
        current_clusters = np.argmin(distances, axis=1)
        tqdm.write(f"距离当前为{np.mean(distances)}")
        if (last_clusters == current_clusters).all():
            tqdm.write("聚类得到的anchors如下")
            areas = clusters[:, 0] * clusters[:, 1]
            sorted_indices = np.argsort(areas)
            sorted_boxes = clusters[sorted_indices]
            with open("config/anchors.txt", "w", encoding="utf-8") as f:
                for w, h in sorted_boxes:
                    f.write(f"{int(w)}, {int(h)}, ")
                f.write("\n")
            break
        for cluster in range(k):
            clusters[cluster] = np.median(boxes[current_clusters == cluster], axis=0)
        last_clusters = current_clusters


def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection + 1e-9)
    return iou_


if __name__ == "__main__":
    main()
