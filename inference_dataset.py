import torch
import torchvision
import torchvision.transforms as T
import cv2
from PIL import Image
import os
from model.SemLA import SemLA
import matplotlib.pyplot as plt
import numpy as np
from einops.einops import rearrange
from model.utils import YCbCr2RGB, RGB2YCrCb, make_matching_figure
import pdb
# Test on a pair of images

MIN_MATCH_COUNT = 5
FLANN_INDEX_KDTREE = 1

def load_imgs(vis_folder, ir_folder):
    vis_imgs = {}
    ir_imgs = {}
    for f in os.listdir(vis_folder):
        img = cv2.imread(os.path.join(vis_folder,f))
        if img is not None:
            vis_imgs[f] = img

    for f in os.listdir(ir_folder):
        img = cv2.imread(os.path.join(ir_folder,f))
        if img is not None:
            ir_imgs[f] = img
    return vis_imgs, ir_imgs


def div_superpixel(img,region_size=20,ruler=10.0,iter=20):
    superpixel = cv2.ximgproc.createSuperpixelSLIC(img,algorithm=cv2.ximgproc.MSLIC, region_size = region_size, ruler = ruler)
    superpixel.iterate(iter)
    super_mask = superpixel.getLabelContourMask()
    temp = img1_raw

    for row in range(temp.shape[0]):
        for col in range(temp.shape[1]):
            if super_mask[row,col] == 255:
                temp[row,col] = 255
    superp_img = Image.fromarray(temp)
    #superp_img.save("./dataset_superpixel_ir/{}.png".format(k))
    return superp_img

def save_feat_map(feat_map):
    trans = T.ToPILImage()
    img = trans(feat_map)
    img.save("feat_sair_map_sample.png")


def coarse_reg_feat(vis_feat, ir_feat):
    norm_vis = torch.clamp(vis_feat,0,1)
    norm_ir = torch.clamp(ir_feat,0,1)
    tensor_vis = norm_vis*255
    tensor_ir = norm_ir*255
    tensor_8U_vis = tensor_vis.to(torch.uint8)
    tensor_8U_ir = tensor_ir.to(torch.uint8)
    numpy_vis = tensor_8U_vis.detach().cpu().numpy()
    numpy_ir = tensor_8U_ir.detach().cpu().numpy()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(numpy_ir,None)
    kp2, des2 = sift.detectAndCompute(numpy_vis,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Filter out poor matches
    good_matches = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good_matches.append(m)
    
    matches = good_matches

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Find homography
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Warp ir to align with vis
    ir_feat_reg = cv2.warpPerspective(ir_feat, H, (vis_feat.shape[1], vis_feat.shape[0]))
    return ir_feat_reg

if __name__ == '__main__':
    # config
    reg_weight_path = "reg.ckpt"
    fusion_weight_path = "fusion75epoch.ckpt"

    vis_folder = "./IR-VIS-Stereo/vis" 
    ir_folder = "./IR-VIS-Stereo/ir"

    vis_dict, ir_dict = load_imgs(vis_folder, ir_folder)
    print("Load Images Successfully! Preparing for matching...")

    match_mode = 'scene' # 'semantic' or 'scene'

    matcher = SemLA()
    # Loading the weights of the registration model
    matcher.load_state_dict(torch.load(reg_weight_path),strict=False)

    # Loading the weights of the fusion model
    matcher.load_state_dict(torch.load(fusion_weight_path), strict=False)

    matcher = matcher.eval().cuda()

    print("Starting to match images...")
    print("--------------------------------------------")
    for k,v in vis_dict.items():
        print("{} Pair Preparing...".format(k))
        img0 = v
        img0_raw = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = ir_dict[k]
        img1_raw = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

        #pdb.set_trace()
        #img1_raw = cv2.cvtColor(img1, cv2.COLO)
        img0_raw = cv2.resize(img0_raw, (640, 480))  # input size shuold be divisible by 8
        img1_raw = cv2.resize(img1_raw, (640, 480))

        img0 = torch.from_numpy(img0_raw)[None].cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.

        img0 = rearrange(img0, 'n h w c ->  n c h w')
        vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img0)
        #pdb.set_trace()
        mkpts0, mkpts1, feat_sa_vi, feat_sa_ir, sa_ir, sa_vi= matcher(vi_Y, img1, matchmode=match_mode)
        #save_feat_map(torch.squeeze(feat_sa_vi))
        #pdb.set_trace()
        #save_feat_map(torch.squeeze(sa_ir))
        #matrix = coarse_reg(torch.squeeze(sa_vi),torch.squeeze(sa_ir))
        #pdb.set_trace()
        mkpts0 = mkpts0.cpu().numpy()
        mkpts1 = mkpts1.cpu().numpy()

        M, prediction = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC,5)
        img1_raw_new = cv2.warpPerspective(img1_raw, M, (img0_raw.shape[1], img1_raw.shape[0]))
        #pdb.set_trace()

        img1_new = torch.from_numpy(img1_raw_new)[None][None].cuda() / 255.
        mkpts0, mkpts1, feat_sa_vi, feat_sa_ir, sa_ir, sa_vi= matcher(vi_Y, img1_new, matchmode=match_mode)
        
        mkpts0 = mkpts0.cpu().numpy()
        mkpts1 = mkpts1.cpu().numpy()

        _, prediction = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC,5)

        prediction = np.array(prediction, dtype=bool).reshape([-1])
        mkpts0_tps = mkpts0[prediction]
        mkpts1_tps = mkpts1[prediction]
        tps = cv2.createThinPlateSplineShapeTransformer()
        #tps.setRegularizationParameter(0.2)
        #pdb.set_trace()
        mkpts0_tps_ransac = mkpts0_tps.reshape(1, -1, 2)
        mkpts1_tps_ransac = mkpts1_tps.reshape(1, -1, 2)

        matches = []
        for j in range(1, mkpts0.shape[0] + 1):
            matches.append(cv2.DMatch(j, j, 0))

        tps.estimateTransformation(mkpts0_tps_ransac, mkpts1_tps_ransac, matches)
        #pdb.set_trace()
        img1_raw_trans = tps.warpImage(img1_raw_new)
        sa_ir = tps.warpImage(sa_ir[0][0].detach().cpu().numpy())
        sa_ir = torch.from_numpy(sa_ir)[None][None].cuda()
        #pdb.set_trace()

        img1_trans = torch.from_numpy(img1_raw_trans)[None][None].cuda() / 255.
        fuse = matcher.fusion(torch.cat((vi_Y, img1_trans), dim=0), sa_ir, matchmode=match_mode).detach()
        #pdb.set_trace()

        fuse = YCbCr2RGB(fuse, vi_Cb, vi_Cr)
        fuse = fuse.detach().cpu()[0]
        fuse = rearrange(fuse, ' c h w ->  h w c').detach().cpu().numpy()

        fig = make_matching_figure(fuse, img0_raw, img1_raw, mkpts0_tps, mkpts1_tps)

        plt.savefig("./dataset_result_semantic/{}.png".format(k))
        print("{} Pair Matched Successfully!".format(k))