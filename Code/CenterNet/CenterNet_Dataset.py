#!/usr/bin/env python3
import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

IMG_H, IMG_W = 720, 720
BOX_W, BOX_H = 100, 100
OUTPUT_SIZE = IMG_H // 4  
DOWN_RATIO = IMG_H // OUTPUT_SIZE
DEBUG_HEATMAP = True  

def gaussian2D(shape, sigma=1):
    m, n = [(s - 1.) / 2. for s in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(hm, center, r):
    d = 2 * r + 1
    g = gaussian2D((d, d), sigma=d / 6)
    x, y = int(center[0]), int(center[1])
    if x < -d // 2 or y < -d // 2 or x >= hm.shape[1] + d // 2 or y >= hm.shape[0] + d // 2:
        return
    left, right = max(0, x - r), min(hm.shape[1], x + r + 1)
    top, bottom = max(0, y - r), min(hm.shape[0], y + r + 1)
    if left >= right or top >= bottom:
        return
    g_left = max(0, -x + r)
    g_top = max(0, -y + r)
    patch = g[g_top:g_top + (bottom - top), g_left:g_left + (right - left)]
    hm[top:bottom, left:right] = np.maximum(hm[top:bottom, left:right], patch)


def create_targets(boxes, debug=False):
    hm = np.zeros((1, OUTPUT_SIZE, OUTPUT_SIZE), np.float32)
    size = np.zeros((2, OUTPUT_SIZE, OUTPUT_SIZE), np.float32)
    off = np.zeros((2, OUTPUT_SIZE, OUTPUT_SIZE), np.float32)
    mask = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), np.float32)
    valid = 0
    for x, y, w, h in boxes:
        if x + w <= 0 or y + h <= 0 or x >= IMG_W or y >= IMG_H:
            continue
        x0, y0 = max(0, x), max(0, y)
        w0 = max(1, min(w, IMG_W - x0))
        h0 = max(1, min(h, IMG_H - y0))
        cx, cy = x0 + w0 / 2, y0 + h0 / 2
        cx_o, cy_o = cx / DOWN_RATIO, cy / DOWN_RATIO
        if not (0 <= cx_o < OUTPUT_SIZE and 0 <= cy_o < OUTPUT_SIZE):
            continue
        xi, yi = int(cx_o), int(cy_o)
        r = max(4, int(min(w0, h0) / (DOWN_RATIO * 2)))
        draw_gaussian(hm[0], (cx_o, cy_o), r)
        size[:, yi, xi] = [w0, h0]
        off[:, yi, xi] = [cx_o - xi, cy_o - yi]
        mask[yi, xi] = 1
        valid += 1
    if debug:
        print(f"create_targets: valid={valid}/{len(boxes)}, hm_max={hm.max():.4f}")
    if DEBUG_HEATMAP:
        os.makedirs('debug', exist_ok=True)
        cv2.imwrite(f'debug/heatmap_{np.random.randint(1e4)}.png', (hm[0]*255).astype(np.uint8))
    return hm, size, off, mask

def augment_image_and_boxes(img, boxes):
    H, W = img.shape
    # 1) Random zoom
    if random.random() < 0.5:
        scale = random.uniform(0.8, 1.2)
        new_h, new_w = int(H*scale), int(W*scale)
        img_zoom = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if scale < 1.0:
            pad_h, pad_w = (H-new_h)//2, (W-new_w)//2
            canvas = np.zeros_like(img)
            canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_zoom
            img = canvas
            for b in boxes:
                b[0] = b[0]*scale + pad_w
                b[1] = b[1]*scale + pad_h
                b[2] *= scale; b[3] *= scale
        else:
            start_h, start_w = (new_h-H)//2, (new_w-W)//2
            img = img_zoom[start_h:start_h+H, start_w:start_w+W]
            for b in boxes:
                b[0] = b[0]*scale - start_w
                b[1] = b[1]*scale - start_h
                b[2] *= scale; b[3] *= scale
    if random.random() < 0.5:
        dx, dy = random.uniform(-10,10), random.uniform(-10,10)
        M = np.array([[1,0,dx],[0,1,dy]],dtype=np.float32)
        img = cv2.warpAffine(img, M, (W,H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        for b in boxes: b[0]+=dx; b[1]+=dy
    if random.random() < 0.5:
        img = np.fliplr(img).copy()
        for b in boxes: b[0] = W - b[0] - b[2]
    if random.random() < 0.5:
        angle = random.uniform(-15,15)
        M = cv2.getRotationMatrix2D((W/2,H/2),angle,1.0)
        img = cv2.warpAffine(img,M,(W,H),borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        for b in boxes:
            cx,cy=b[0]+b[2]/2,b[1]+b[3]/2
            nx,ny = M[0,0]*cx+M[0,1]*cy+M[0,2], M[1,0]*cx+M[1,1]*cy+M[1,2]
            b[0],b[1] = nx-b[2]/2, ny-b[3]/2
    if random.random()<0.5:
        clahe = cv2.createCLAHE(2.0,(8,8)); img=clahe.apply(img)
    if random.random()<0.5:
        img=np.clip(img*random.uniform(0.8,1.2)+random.uniform(-20,20),0,255).astype(np.uint8)
    if random.random()<0.5:
        noise=np.random.normal(0,5,img.shape); img=np.clip(img+noise,0,255).astype(np.uint8)
    return img, boxes

class MotorDataset(Dataset):
    def __init__(self,csv_path=None,img_dir=None,debug=False,transform=None):
        base = os.path.dirname(__file__);
        root = os.path.abspath(os.path.join(base,'..','..'))
        self.csv_path=csv_path or os.path.join(root,'Dataset','train_labels.csv')
        self.img_dir =img_dir or os.path.join(root,'Dataset','train')
        self.debug   =debug
        self.transform=transform or augment_image_and_boxes
        print(f"CSV path: {self.csv_path}")
        print(f"Image directory: {self.img_dir}")
        df=pd.read_csv(self.csv_path)
        self.samples=[]
        for _,row in df.iterrows():
            tid=str(row['tomo_id']);cx,cy=row['Motor axis 1'],row['Motor axis 2']
            if not isinstance(cx,(int,float)) or not isinstance(cy,(int,float)): continue
            box=[cx-BOX_W/2,cy-BOX_H/2,BOX_W,BOX_H]
            folder=os.path.join(self.img_dir,tid)
            if not os.path.isdir(folder): continue
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(('.png','.jpg')):
                    self.samples.append({'path':os.path.join(folder,fn),'boxes':[box]}); break
        if self.debug: print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        rec=self.samples[idx]
        img=cv2.imread(rec['path'],cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError(rec['path'])
        boxes=[b.copy() for b in rec['boxes']]
        orig_img,orig_boxes=img.copy(),[b.copy() for b in boxes]
        img,boxes=self.transform(img,boxes)
        cx,cy=boxes[0][0]+boxes[0][2]/2,boxes[0][1]+boxes[0][3]/2
        if not (0<=cx<IMG_W and 0<=cy<IMG_H):
            if self.debug: print(f"OOB revert ({cx:.1f},{cy:.1f})")
            img,boxes=orig_img,orig_boxes
        orig_h,orig_w=img.shape; sx,sy=IMG_W/orig_w,IMG_H/orig_h
        boxes=[[b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy] for b in boxes]
        img=cv2.resize(img,(IMG_W,IMG_H)).astype(np.float32)/255.0
        mu,sig=img.mean(),img.std(); img=(img-mu)/(sig+1e-6); img=np.clip(img,-3,3)
        if random.random()<0.5:
            x0,y0,w0,h0=boxes[0];cx,cy=x0+w0/2,y0+h0/2;pad=int(BOX_W*0.2)
            x1,y1=max(0,int(cx-BOX_W-pad)),max(0,int(cy-BOX_H-pad))
            x2,y2=min(IMG_W,int(cx+BOX_W+pad)),min(IMG_H,int(cy+BOX_H+pad))
            crop=img[y1:y2,x1:x2]; boxes[0][0]-=x1;boxes[0][1]-=y1
            img=cv2.resize(crop,(IMG_W,IMG_H),interpolation=cv2.INTER_LINEAR)
        if random.random()<0.3: img=cv2.GaussianBlur(img,(3,3),0)
        if random.random()<0.2:
            cxr,cyr=random.randint(0,IMG_W-1),random.randint(0,IMG_H-1);l=BOX_W//2
            x0r,y0r=max(0,cxr-l),max(0,cyr-l);x1r,y1r=min(IMG_W,cxr+l),min(IMG_H,cyr+l)
            img[y0r:y1r,x0r:x1r]=img.mean()
        hm,sz,off,mask=create_targets(boxes,debug=self.debug and idx==0)
        img_t=torch.tensor(np.repeat(img[None],3,axis=0),dtype=torch.float32)
        ctr={'heatmap':torch.tensor(hm), 'size':torch.tensor(sz), 'offset':torch.tensor(off), 'mask':torch.tensor(mask)}
        xyxy=[]
        for x,y,w,h in boxes: xyxy.append([max(0,x),max(0,y),min(IMG_W,x+w),min(IMG_H,y+h)])
        det={'boxes':torch.tensor(xyxy,dtype=torch.float32), 'labels':torch.ones((len(xyxy),),dtype=torch.int64)}
        return img_t,ctr,det

if __name__=='__main__':
    ds=MotorDataset(debug=True)
    os.makedirs('debug/vis',exist_ok=True)
    for i in range(min(3,len(ds))):
        rec=ds.samples[i]
        raw=cv2.imread(rec['path'],cv2.IMREAD_GRAYSCALE)
        raw_r=cv2.resize(raw,(IMG_W,IMG_H))
        orig_boxes=[b.copy() for b in rec['boxes']]
        aug_img,aug_boxes=ds.transform(raw.copy(),orig_boxes)
        aug_r=cv2.resize(aug_img,(IMG_W,IMG_H))
        hm,_,_,_=create_targets(aug_boxes,debug=False)
        fig,ax=plt.subplots(1,3,figsize=(12,4))
        ax[0].imshow(raw_r,cmap='gray');x,y,w,h=orig_boxes[0]
        ax[0].add_patch(plt.Rectangle((x*IMG_W/raw.shape[1],y*IMG_H/raw.shape[0]),w*IMG_W/raw.shape[1],h*IMG_H/raw.shape[0],edgecolor='r',fill=False,linewidth=2));ax[0].axis('off')
        ax[1].imshow(aug_r,cmap='gray');bx=aug_boxes[0]
        ax[1].add_patch(plt.Rectangle((bx[0],bx[1]),bx[2],bx[3],edgecolor='g',fill=False,linewidth=2));ax[1].axis('off')
        ax[2].imshow(hm[0],cmap='hot');ax[2].axis('off')
        fig.tight_layout();fig.savefig(f'debug/vis/sample_{i}.png');plt.close(fig)
        print(f"Saved visualization sample_{i}.png")
