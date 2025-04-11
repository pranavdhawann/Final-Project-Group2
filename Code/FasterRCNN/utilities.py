import pandas as pd

def get_frcnn_annotations(train_labels_path):

    bb_w, bb_h = 24,24

    df = pd.read_csv(train_labels_path)
    grouped_list = [group for _, group in df.groupby(["tomo_id", "Motor axis 0"])]
    annotations = []     # {"filename", "boxes", labels}

    for group in grouped_list:
        file_name = group.iloc[0]["tomo_id"]+"/slice_"+str(int(group.iloc[0]["Motor axis 0"])).zfill(4)+".jpg" \
            if group.iloc[0]["Number of motors"] != 0 \
            else group.iloc[0]["tomo_id"]+"/slice_"+str(int(group.iloc[0]["Array shape (axis 0)"]/2)).zfill(4)+".jpg"
        boxes = []
        labels = []
        if group.iloc[0]["Number of motors"] != 0:
            for _, row in group.iterrows():
                cy = row["Motor axis 1"]
                cx = row["Motor axis 2"]

                x0,y0 = max(0, cx-bb_w/2), max(0, cy-bb_h/2)
                x1,y1 = min(row["Array shape (axis 2)"], cx+bb_w/2), min(row["Array shape (axis 1)"], cy+bb_h/2)

                x0,x1 = x0/row["Array shape (axis 2)"], x1/row["Array shape (axis 2)"]
                y0,y1 = y0/row["Array shape (axis 1)"], y1/row["Array shape (axis 1)"]

                boxes.append([x0,y0,x1,y1])
                labels.append("motor")
        annotations.append(
            {
                "file_name": file_name,
                "boxes": boxes,
                "labels": labels,
            }
        )
    return annotations