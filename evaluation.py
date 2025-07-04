import torch
import tqdm


def val(model, val_loader, loss_fn, device, writer=None):
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        total_samples = 0
        gloabl_step = 0
        with tqdm.tqdm(val_loader, total=len(val_loader)) as pbar:
            for i, batch in enumerate(pbar):
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = [label.to(device) for label in batch_y]
                outputs = model(batch_x)[1]
                loss = loss_fn(outputs, batch_y)['original_loss'].item()
                batch_size = batch_x.size(0)
                epoch_loss += loss * batch_size
                total_samples += batch_size
                avg_loss = epoch_loss / total_samples
                # writer.add_scalars("loss", {'val_loss': avg_loss}, gloabl_step)
                gloabl_step += 1
                pbar.set_postfix(**{"val_loss": f"{avg_loss:.4f}"})

if __name__ == '__main__':
    from models.yolo import Model
    from utils.general import check_yaml
    from utils.ComputeLoss import YOLOv3LOSS
    from utils.dataloader import YOLODataset, yolo_collate_fn
    from torch.utils.data.dataloader import DataLoader

    model = Model(check_yaml("yolov3-tiny.yaml"))
    model.load_state_dict(torch.load(r'0.5082_best_61.pth', map_location=torch.device('cpu'))['model'])

    val_loader = DataLoader(YOLODataset(dataset_type='voc', Train=False), batch_size=4, shuffle=False, collate_fn=yolo_collate_fn)

    val(model, val_loader, YOLOv3LOSS(model), torch.device('cpu'))