import os
import torch
import numpy as np
from PIL import Image
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
import torchvision.transforms as Transforms
from torchvision.models.inception import inception_v3

import VITON.Parser_Based.HR_VITON.eval_models.evals_model as models
from VITON.Parser_Based.HR_VITON.eval_models.evals_model import calculate_kid, calculate_fid
ground_truth_dir, predict_dir, ground_truth_mask_dir = None, None, None


def calculate_psnr(img1, img2, max_pixel=255.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def Evaluation(opt, pred_list, gt_list, gt_mask_list):
    global ground_truth_dir, predict_dir, ground_truth_mask_dir
    T1 = Transforms.ToTensor()
    T_mask = Transforms.Compose([Transforms.ToTensor(),Transforms.Normalize((0.5), (0.5))])
    T2 = Transforms.Compose([Transforms.Resize((128, 128)),
                            Transforms.ToTensor(),
                            Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                 std=(0.5, 0.5, 0.5))])
    T3 = Transforms.Compose([Transforms.Resize((299, 299)),
                            Transforms.ToTensor(),
                            Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                 std=(0.5, 0.5, 0.5))])
    T4 = Transforms.Compose([Transforms.ToTensor(), Transforms.Normalize((0.5), (0.5), (0.5))])
    splits = 1 # Hyper-parameter for IS score

    model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True)
    model.eval()
    inception_model = inception_v3(pretrained=True, transform_input=False).type(torch.cuda.FloatTensor)
    inception_model.eval()

    avg_ssim, avg_mse, avg_distance,avg_psnr = 0.0, 0.0, 0.0,0.0
    preds = np.zeros((len(gt_list), 1000))
    lpips_list = []
    
    # Comute features vectors for FID and KID
    features_real = []
    features_fake = []
    with torch.no_grad():
        print("Calculate SSIM, MSE, LPIPS, KID, FID, PSNR...")
        for i, img_pred in enumerate(pred_list):
            # Calculate SSIM
            gt_img = Image.open(os.path.join(ground_truth_dir, img_pred))
            gt_np = np.asarray(gt_img.convert('L'))
            
            pred_img = Image.open(os.path.join(predict_dir, img_pred))
            pred_np = np.asarray(pred_img.convert('L'))
            
            pred_mask_img = Image.open(os.path.join(ground_truth_mask_dir, img_pred))
            pred_mask_np = np.asarray(pred_mask_img.convert('L'))
            binary_image = np.where(pred_mask_np > 127, 255, 0).astype(np.uint8)

            pred_img_tensor = T4(pred_img)
            pred_img_mask_tensor = T4(pred_mask_img)

            if opt.clip_warping:
                # Apply the formula
                result_img_tensor = pred_img_tensor * pred_img_mask_tensor + torch.ones_like(pred_img_tensor) * (1 - pred_img_mask_tensor)
                result_img_tensor.clamp_(0, 1)
                
                result_img_numpy = result_img_tensor.permute(1,2,0).numpy()

                # The numpy array has to be converted from float [0, 1] range to an 8-bit format
                result_img_numpy = (result_img_numpy * 255).astype(np.uint8)

                # Convert the numpy array to a PIL Image
                pred_img = Image.fromarray(result_img_numpy)
        
            
            pred_img_IS = T3(pred_img).unsqueeze(0).cuda()
            gt_img_LPIPS = T2(gt_img).unsqueeze(0).cuda()
            pred_img_LPIPS = T2(pred_img).unsqueeze(0).cuda()
            
            
            assert gt_img.size == pred_img.size, f"{gt_img.size} vs {pred_img.size}"
            avg_ssim += ssim(gt_np, pred_np, data_range=255, gaussian_weights=True, use_sample_covariance=False)

            # Calculate LPIPS
            lpips_list.append((img_pred, model.forward(gt_img_LPIPS, pred_img_LPIPS).item()))
            avg_distance += lpips_list[-1][1]
            # Calculate Inception model prediction
            preds[i] = F.softmax(inception_model(pred_img_IS)).data.cpu().numpy()

            gt_img_MSE = T1(gt_img).unsqueeze(0).cuda()
            pred_img_MSE = T1(pred_img).unsqueeze(0).cuda()
            avg_mse += F.mse_loss(gt_img_MSE, pred_img_MSE)
            
            
            pred_img_np = np.array(pred_img)
            gt_img_np = np.array(gt_img)
            avg_psnr += calculate_psnr(pred_img_np, gt_img_np)
            
            # Calculate features for FID and KID
            pred_img_feature = F.softmax(inception_model(pred_img_IS)).data.cpu().numpy().flatten()
            gt_img_feature = F.softmax(inception_model(T3(gt_img).unsqueeze(0).cuda())).data.cpu().numpy().flatten()
            features_fake.append(pred_img_feature)
            features_real.append(gt_img_feature)
            print(f"step: {i+1} evaluation... lpips:{lpips_list[-1][1]}")

        avg_ssim /= len(gt_list)
        avg_mse = avg_mse / len(gt_list)
        avg_psnr /= len(gt_list)
        avg_distance = avg_distance / len(gt_list)
        features_real = np.array(features_real)
        features_fake = np.array(features_fake)
        fid_value = calculate_fid(features_real, features_fake)
        kid_value = calculate_kid(features_real, features_fake)
        # Calculate Inception Score
        split_scores = [] # Now compute the mean kl-divergence

        lpips_list.sort(key=lambda x: x[1], reverse=True)
        for name, score in lpips_list:
            f = open(os.path.join(opt.results_dir, 'lpips.txt'), 'a')
            f.write(f"{name} {score}\n")
            f.close()
        print("Calculate Inception Score...")
        for k in range(splits):
            part = preds[k * (len(gt_list) // splits): (k+1) * (len(gt_list) // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        IS_mean, IS_std = np.mean(split_scores), np.std(split_scores)
    f = open(os.path.join(opt.results_dir, 'eval.txt'), 'a')
    f.write(f"SSIM : {avg_ssim} / PSNR: {avg_psnr} / MSE : {avg_mse} / LPIPS : {avg_distance} / FID: {fid_value} / KID: {kid_value} \n")
    f.write(f"IS_mean : {IS_mean} / IS_std : {IS_std}\n")
    
    f.close()
    return avg_ssim, avg_mse, avg_distance, avg_psnr, IS_mean, IS_std, fid_value, kid_value

def evaluate(opt):
    global ground_truth_dir, predict_dir, ground_truth_mask_dir
    # prediction_list = os.listdir(os.path.join(opt.results_dir, 'prediction'))
    ground_truth_dir = os.path.join(opt.results_dir, 'ground_truth')
    ground_truth_mask_dir = os.path.join(opt.results_dir, 'ground_truth_mask')
    predict_dir = os.path.join(opt.results_dir, 'prediction')
    
    ground_truth_list = os.listdir(ground_truth_dir)
    ground_truth_mask_list = os.listdir(ground_truth_mask_dir)

    prediction_list = os.listdir(predict_dir)
    prediction_list.sort()
    ground_truth_list.sort()
    
    avg_ssim, avg_mse, avg_distance, avg_psnr, IS_mean, IS_std, fid_value, kid_value= Evaluation(opt, prediction_list, ground_truth_list, ground_truth_mask_list)
    print("SSIM : %f / MSE : %f / LPIPS : %f / FID: %f / KID: %f / PSNR: %f" % (avg_ssim, avg_mse, avg_distance, fid_value, kid_value, avg_psnr))
    print("IS_mean : %f / IS_std : %f" % (IS_mean, IS_std))