import torch, cv2, os
import numpy as np

def mu_tonemap(img):
    MU = 5000.0
    return torch.log(1.0 + MU * (img+1.0)/2.0) / np.log(1.0 + MU)

def write_hdr(hdr_image, path):
	
    # Writing HDR image in radiance format  
	
    norm_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
    with open(path, "wb") as f:
        # print('max')
        # print('max')
        # print(norm_image.max())
        # print(norm_image.min())
        norm_image = (norm_image - norm_image.min())/(norm_image.max() - norm_image.min())  # normalisation function
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(norm_image.shape[0], norm_image.shape[1]))
        brightest = np.maximum(np.maximum(norm_image[...,0], norm_image[...,1]), norm_image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((norm_image.shape[0], norm_image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(norm_image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)
        rgbe.flatten().tofile(f)
        f.close()

def save_hdr_image(img_tensor, batch, path):
    img = img_tensor.data[batch].cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0))

    if img.shape[2] == 1 or img.shape[2] > 3:        
        img = img[:,:,0]
    
    write_hdr(img.astype(np.float32), path)

def save_ldr_image(img_tensor, batch, path):
    img = img_tensor.data[batch].cpu().float().numpy()
    img = 255 * np.transpose(img, (1, 2, 0))

    if img.shape[2] == 1 or img.shape[2] > 3:        
        img = img[:,:,0]

    # img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def save_checkpoint(epoch, model):
    checkpoint_path = os.path.join('./checkpoints', 'epoch_' + str(epoch) + '.ckpt')
    latest_path = os.path.join('./checkpoints', 'latest.ckpt')
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(model.state_dict(), latest_path)
    np.savetxt('./checkpoints/state.txt', [epoch + 1], fmt='%d')
    print('Saved checkpoint for epoch ', epoch)

def update_lr(optimizer, epoch, opt):
    new_lr = opt.lr - opt.lr * (epoch - opt.lr_decay_after)/(opt.epochs - opt.lr_decay_after)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    print('Learning rate decayed. Updated LR is: %.6f'%new_lr)

