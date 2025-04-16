import torch
from torch.utils.data import DataLoader
from centernet.model import CenterNet
from centernet.dataset import MotorDataset
from centernet.loss import focal_loss, l1_loss

model = CenterNet().cuda()
dataset = MotorDataset("data/images", "data/annotations.json")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    total_loss = 0

    for batch in dataloader:
        images = batch["image"].cuda()
        heatmaps = batch["heatmap"].cuda()
        sizes = batch["size"].cuda()
        offsets = batch["offset"].cuda()
        mask = batch["mask"].cuda().unsqueeze(1)

        outputs = model(images)
        loss_hm = focal_loss(outputs["heatmap"], heatmaps)
        loss_sz = l1_loss(outputs["size"], sizes, mask)
        loss_of = l1_loss(outputs["offset"], offsets, mask)

        loss = loss_hm + 0.1 * loss_sz + loss_of

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/100 - Loss: {total_loss:.4f}")