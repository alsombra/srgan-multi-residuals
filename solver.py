import torch
import torch.nn as nn
import os
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from model import *
import numpy as np
from data_loader import Scaling, Scaling01, ImageFolder, random_downscale, bicub_downscale, bilin_downscale, nearest_downscale
from utils import Kernels, load_kernels
from PIL import Image,ImageDraw


torch.set_default_tensor_type(torch.FloatTensor)


class Solver(object):
    def __init__(self, data_loader, config):
        # Data loader
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)

        # Model hyper-parameters
        self.num_channels = config.num_channels  # num_channels = 6
        self.scale_factor = config.scale_factor  # scale_factor = 2

        # Training settings
        self.total_step = config.total_step # 50000
        self.content_loss_function = config.content_loss_function
        self.residual_loss_function = config.residual_loss_function
        self.lr = config.lr 
        self.beta1 = config.beta1 # 0.5  ????????????????? testar 0.9 ou 0.001
        self.beta2 = config.beta2 # 0.99  ???????????????
        self.trained_model = config.trained_model
        self.trained_discriminator = config.trained_discriminator
        self.use_tensorboard = config.use_tensorboard
        self.start_step = -1
        
        #Test settings
        self.test_mode = config.test_mode
        self.test_image_path = config.test_image_path
        self.evaluation_step = config.evaluation_step
        self.evaluation_size = config.evaluation_size
        
        # Path and step size
        self.log_path = config.log_path
        self.result_path = config.result_path
        self.model_save_path = config.model_save_path
        self.log_step = config.log_step # log_step = 10
        self.sample_step = config.sample_step # sample_step = 100
        self.model_save_step = config.model_save_step # model_save_step = 1000

        # Device configuration
        self.device = config.device

        # Initialize model
        self.hr_shape = (config.image_size * self.scale_factor, config.image_size * self.scale_factor)

        self.build_model(config) 
        
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.trained_model:
            self.load('generator')
            print('loaded trained model (step: {})..!'.format(self.trained_model.split('.')[0])) 
            
        if self.trained_discriminator:
            self.load('discriminator')
            print('loaded trained discriminator (step: {})..!'.format(self.trained_discriminator.split('.')[0])) 

    def build_model(self, config):
        # model and optimizer
        self.model = GeneratorResNet(in_channels=self.num_channels)  #OBS: model = generator
        self.discriminator = Discriminator(input_shape=(3, *self.hr_shape)) # 3 channels residual - RGB 
        self.feature_extractor = FeatureExtractor()
        # Set feature extractor to inference mode
        self.feature_extractor.eval()
    
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.model.parameters(), self.lr, [self.beta1, self.beta2])
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), self.lr, [self.beta1, self.beta2])
        
        self.model.to(self.device)
        self.discriminator.to(self.device)
        self.feature_extractor.to(self.device)

    def load(self, type_of_net):
        if type_of_net == 'generator':
            filename = os.path.join(self.model_save_path, '{}'.format(self.trained_model))
            S = torch.load(filename)
            self.model.load_state_dict(S['SR'])
            try:
                self.optimizer_G.load_state_dict(S['optimizer_state_dict'])                       
            except KeyError as error:
                print('There is no '+str(error)+' in loaded model. Loading model without optimizer_params')
            try:
                self.start_step = S['epoch'] - 1
            except KeyError as error:
                print('There is no '+str(error)+' in loaded model. Loading model without epoch info')
        if type_of_net=='discriminator':
            filename = os.path.join(self.model_save_path, '{}'.format(self.trained_discriminator))
            S = torch.load(filename)
            self.discriminator.load_state_dict(S['SR'])
            try:
                self.optimizer_D.load_state_dict(S['optimizer_state_dict'])  
            except KeyError as error:
                print('There is no '+str(error)+' in loaded model. Loading model without optimizer_params')

    def update_lr(self, lr):
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

    def detach(self, x):  # NOT USED. To learn more SEE https://pytorch.org/blog/pytorch-0_4_0-migration-guide/
        return x.data
    #############################################################
    
    def get_hr_hat_from_resid(self, lr_images, reconsts):
        tmp1 = lr_images.data.cpu().numpy().transpose(0,2,3,1)*255
        image_list = [np.array(Image.fromarray(tmp1.astype(np.uint8)[i]).resize((128,128), Image.BICUBIC)) for i in range(len(lr_images))]
        images_hr_bicubic= np.stack(image_list)
        #return this ^
        images_hr_bicubic = images_hr_bicubic.transpose(0,3,1,2)
        images_hr_bicubic = Scaling(images_hr_bicubic)
        images_hr_bicubic = torch.from_numpy(images_hr_bicubic).float().to(self.device) # NUMPY to TORCH
        hr_images_hat = reconsts + images_hr_bicubic
        hr_images_hat = hr_images_hat.data.cpu().numpy()
        hr_images_hat = Scaling01(hr_images_hat)
        hr_images_hat = torch.from_numpy(hr_images_hat).float().to(self.device) # NUMPY to TORCH

        return hr_images_hat
        
    
    def get_trio_images(self, lr_image,hr_image, reconst):
        tmp1 = lr_image.data.cpu().numpy().transpose(0,2,3,1)*255
        image_list = [np.array(Image.fromarray(tmp1.astype(np.uint8)[i]).resize((128,128), Image.BICUBIC)) for i in range(self.data_loader.batch_size)]
        image_hr_bicubic= np.stack(image_list)
        image_hr_bicubic_single = np.squeeze(image_hr_bicubic)
        print('hr_bicubic_single:', image_hr_bicubic_single.shape)
        #return this ^
        image_hr_bicubic = image_hr_bicubic.transpose(0,3,1,2)
        image_hr_bicubic = Scaling(image_hr_bicubic)
        image_hr_bicubic = torch.from_numpy(image_hr_bicubic).float().to(self.device) # NUMPY to TORCH
        hr_image_hat = reconst + image_hr_bicubic
        hr_image_hat = hr_image_hat.data.cpu().numpy()
        hr_image_hat = Scaling01(hr_image_hat)
        hr_image_hat = np.squeeze(hr_image_hat).transpose((1, 2, 0))
        hr_image_hat = (hr_image_hat*255).astype(np.uint8)
        print('hr_image_hat : ', hr_image_hat.shape)
        #return this ^
        hr_image = hr_image.data.cpu().numpy().transpose(0,2,3,1)*255
        hr_image = np.squeeze(hr_image.astype(np.uint8))
        #return this ^
        return Image.fromarray(image_hr_bicubic_single), Image.fromarray(hr_image_hat), Image.fromarray(hr_image)

    def create_grid(self, lr_image,hr_image, reconst):
        'generate grid image: LR Image | HR image Hat (from model) | HR image (original)'
        'lr_image = lr_image tensor from dataloader (can be batch)'
        'hr_image = hr_image tensor from dataloader (can be batch)'
        'reconst = output of model (HR residual)'
        tmp1 = lr_image.data.cpu().numpy().transpose(0,2,3,1)*255
        image_list = [np.array(Image.fromarray(tmp1.astype(np.uint8)[i]).resize((128,128), Image.BICUBIC)) for i in range(self.data_loader.batch_size)]
        image_hr_bicubic= np.stack(image_list).transpose(0,3,1,2)
        image_hr_bicubic = Scaling(image_hr_bicubic)
        image_hr_bicubic = torch.from_numpy(image_hr_bicubic).float().to(self.device) # NUMPY to TORCH
        hr_image_hat = reconst + image_hr_bicubic
                
        hr_image_hat = hr_image_hat.data.cpu().numpy()
        hr_image_hat = Scaling01(hr_image_hat)
        hr_image_hat = torch.from_numpy(hr_image_hat).float().to(self.device) # NUMPY to TORCH

        pairs = torch.cat((image_hr_bicubic.data, \
                                hr_image_hat.data,\
                                hr_image.data), dim=3)
        grid = make_grid(pairs, 1) 
        tmp = np.squeeze(grid.cpu().numpy().transpose((1, 2, 0)))
        grid = (255 * tmp).astype(np.uint8)
        return grid
    
    def img_add_info(self, img_paths, img, step, loss):
        'receives tensor as img'
        added_text = Image.new('RGB', (500, img.shape[0]), color = 'white')
        d = ImageDraw.Draw(added_text)
        d.text((10,10), "model trained for {} steps, loss G: (comparing residuals): {:.4f}".format(step, loss.item()) + \
               "\n" + '\n'.join([os.path.basename(path) for path in img_paths]), fill='black')
        imgs_comb = np.hstack((np.array(img), added_text))
        
        d.text((10,10), "model trained for {} steps, loss G: (comparing residuals): {:.4f}".format(step, loss.item()), fill='black')
        
        imgs_comb = Image.fromarray(imgs_comb)
        return imgs_comb    
    

    def train(self):
        self.model.train()
        self.discriminator.train()
        
        
#         cuda = torch.cuda.is_available()    
#         Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor    
    
            # GANs Loss 
        criterion_GAN = nn.MSELoss()

        # Residual Loss
        if self.residual_loss_function == 'l1':
            criterion_resid = nn.L1Loss()
        if self.residual_loss_function == 'l2':
            criterion_resid = nn.MSELoss()
                
        # Content Loss
        if self.content_loss_function == 'l1':
            criterion_content = nn.L1Loss()
        elif self.content_loss_function == 'l2':
            criterion_content = nn.MSELoss()


        # Data iter
        data_iter = iter(self.data_loader)
        iter_per_epoch = len(self.data_loader)
        
        #Initialize steps
        start = self.start_step + 1 # if not loading trained start = 0     
                    
        for step in range(self.start_step, self.total_step):
            
            self.model.train() # adicionei pq o cara no fim (p/ samples) colocou modo eval() e esqueceu de voltar

            # Reset data_iter for each epoch  
            if (step+1) % iter_per_epoch == 0:     
                data_iter = iter(self.data_loader)  
            
            img_paths, lr_images, hr_images, x, y = next(data_iter)
            lr_images, hr_images, x, y = lr_images.to(self.device), hr_images.to(self.device), x.to(self.device), y.to(self.device)


            # Adversarial ground truths
            valid = np.ones((lr_images.size(0), *self.discriminator.output_shape))
            valid = torch.from_numpy(valid).float().to(self.device)
            fake = np.zeros((lr_images.size(0), *self.discriminator.output_shape))
            fake = torch.from_numpy(fake).float().to(self.device)
               
            # ------------------
            #  Train Generators
            # ------------------
            self.optimizer_G.zero_grad()
            
            out = self.model(x)
            
            # Adversarial Loss
            loss_GAN = criterion_GAN(self.discriminator(out), valid)
            
            #Residual loss
            if self.residual_loss_function !=None:
                loss_resid = criterion_resid(out, y)

            # Content Loss
            #------ get h_hat from out ----
            
            if self.content_loss_function !=None:
                hr_hats = self.get_hr_hat_from_resid(lr_images, out)

                gen_features = self.feature_extractor(hr_hats) #########
                real_features = self.feature_extractor(hr_images)
                loss_content = criterion_content(gen_features, real_features.detach())

                # Total loss loss_G = 1e-3*loss_GAN + loss_content + loss_resid
            loss_G = 1e-3 * loss_GAN
            
            if self.content_loss_function != None:
                loss_G = loss_G + loss_content 
            if self.residual_loss_function != None:
                loss_G = loss_G + loss_resid
            
            # For decoder
            loss_G.backward()

            self.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
 
            self.optimizer_D.zero_grad()
            # Loss of real and fake *RESIDUAL* images
            
            loss_real = criterion_GAN(self.discriminator(y), valid)
            loss_fake = criterion_GAN(self.discriminator(out.detach()), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2
        
            loss_D.backward()
            self.optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Print out log info
            if (step+1) % self.log_step == 0:
                print("[{}/{}] [D loss: {:.5f}] [G loss: {:.5f}]".format(step+1, self.total_step, loss_D.item(), loss_G.item()))

              # Sample images

            if (step+1) % self.sample_step == 0:
                self.model.eval()
                reconst = self.model(x)
                tmp = self.create_grid(lr_images,hr_images, reconst)
                imgs_comb = self.img_add_info(img_paths, tmp, step+1, loss_G)                
                #from IPython.display import display
                grid_PIL = imgs_comb
                grid_PIL.save('./samples/test_{}.jpg'.format(step + 1))
                if self.data_loader.batch_size == 1: #only saves separate images if batch == 1
                    lr_image_np = lr_image.data.cpu().numpy().transpose(0,2,3,1)*255
                    lr_image_np = Image.fromarray(np.squeeze(lr_image_np).astype(np.uint8))
                    hr_bic, hr_hat, hr = self.get_trio_images(lr_image,hr_image, reconst)
                    random_number = np.random.rand(1)[0]
                    lr_image_np.save('./samples/test_{}_lr.png'.format(step + 1))
                    hr_bic.save('./samples/test_{}_hr_bic.png'.format(step + 1))
                    hr_hat.save('./samples/test_{}_hr_hat.png'.format(step + 1))
                    hr.save('./samples/test_{}_hr.png'.format(step + 1))

            # Save check points
            if (step+1) % self.model_save_step == 0:                
                self.save('generator', step+1, loss_G.item(), os.path.join(self.model_save_path, '{}.pth.tar'.format(str(step+1) + '_cont_'+str(self.content_loss_function).upper() + '_resid_' + str(self.residual_loss_function).upper() + '_GEN_' + '_loss_gen_' + "%.3f" % loss_G.item() + '_loss_disc_'+ "%.3f" % loss_D.item())))
                self.save('discriminator', step+1, loss_D.item(), os.path.join(self.model_save_path, '{}.pth.tar'.format(str(step+1) + '_cont_'+str(self.content_loss_function).upper() + '_resid_' + str(self.residual_loss_function).upper() + '_DISC_'+ '_loss_gen_' + "%.3f" % loss_G.item() + '_loss_disc_'+ "%.3f" % loss_D.item())))


    def save(self, type_of_net, step, current_loss, filename):
        if type_of_net == 'generator':
            torch.save({
                'epoch': step+1,
                'SR': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer_G.state_dict(),
                'loss': str(current_loss)
                }, filename)
        elif type_of_net == 'discriminator':
            torch.save({
                'epoch': step+1,
                'SR': self.discriminator.state_dict(),
                'optimizer_state_dict': self.optimizer_D.state_dict(),
                'loss': str(current_loss)
                }, filename)

    def test_and_error(self): #receives batch from dataloader
        'You run it for a random batch from test_set. You can change batch_size for len(test_set)'
        self.model.eval()
        step = self.start_step + 1 # if not loading trained start = 0 
            
        # GANs Loss 
        criterion_GAN = nn.MSELoss()

        # Residual Loss
        if self.residual_loss_function == 'l1':
            criterion_resid = nn.L1Loss()
        if self.residual_loss_function == 'l2':
            criterion_resid = nn.MSELoss()
                
        # Content Loss
        if self.content_loss_function == 'l1':
            criterion_content = nn.L1Loss()
        elif self.content_loss_function == 'l2':
            criterion_content = nn.MSELoss()

        # Data iter
        img_paths, lr_images, hr_images, x, y = next(self.data_iter)
        lr_images, hr_images, x, y = lr_images.to(self.device), hr_images.to(self.device), x.to(self.device), y.to(self.device)


        # Adversarial ground truths
        valid = np.ones((lr_images.size(0), *self.discriminator.output_shape))
        valid = torch.from_numpy(valid).float().to(self.device)
        fake = np.zeros((lr_images.size(0), *self.discriminator.output_shape))
        fake = torch.from_numpy(fake).float().to(self.device)
        
    
        out = self.model(x)
            
        # Adversarial Loss
        loss_GAN = criterion_GAN(self.discriminator(out), valid)
            
        #Residual loss
        if self.residual_loss_function !=None:
            loss_resid = criterion_resid(out, y)

        # Content Loss
        #------ get h_hat from out ----
            
        if self.content_loss_function !=None:
            hr_hats = self.get_hr_hat_from_resid(lr_images, out)

            gen_features = self.feature_extractor(hr_hats) #########
            real_features = self.feature_extractor(hr_images)
            loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss loss_G = 1e-3*loss_GAN + loss_content + loss_resid
        loss_G = 1e-3 * loss_GAN
            
        if self.content_loss_function != None:
            loss_G = loss_G + loss_content 
        if self.residual_loss_function != None:
            loss_G = loss_G + loss_resid

        #Discriminator LOSS
            
        loss_real = criterion_GAN(self.discriminator(y), valid)
        loss_fake = criterion_GAN(self.discriminator(out.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        # Print out log info
        if (step+1) % self.log_step == 0:
            print("Model trained for {} steps, [D loss: {:.5f}] [G loss: {:.5f}]".format(step+1, loss_D.item(), loss_G.item()))
        
        tmp = self.create_grid(lr_images, hr_images, out)
        grid_PIL = self.img_add_info(img_paths, tmp, step, loss_G)
        random_number = np.random.rand(1)[0]
        if self.data_loader.batch_size > 1:
            grid_PIL.save('./test_results/multi_pics_{:.3f}_grid_{}.png'.format(random_number, self.start_step + 1))
            
        elif self.data_loader.batch_size == 1: #only saves separate images if batch == 1
            grid_PIL.save('./results/grids/'+ os.path.basename(img_paths[0])+'_grid_{}.png'.format(self.start_step + 1))
            hr_bic, hr_hat, hr = self.get_trio_images(lr_images,hr_images, out)

            lr_image_np = lr_images.data.cpu().numpy().transpose(0,2,3,1)*255
            lr_image_np = Image.fromarray(np.squeeze(lr_image_np).astype(np.uint8))

            lr_image_np.save('./results/LR_images_snapshot/'+ os.path.basename(img_paths[0])+'_lr_{}.png'.format(self.start_step + 1))
            hr_bic.save('./results/HR_bicub_images/'+ os.path.basename(img_paths[0])+'_hr_bic_{}.png'.format(self.start_step + 1))
            hr_hat.save('./results/HR_HAT_images/'+ os.path.basename(img_paths[0])+'_hr_hat_{}.png'.format(self.start_step + 1))
            hr.save('./results/HR_images/'+ os.path.basename(img_paths[0])+'_hr_{}.png'.format(self.start_step + 1))

    def evaluate(self):
        if self.evaluation_size == -1:
            self.evaluation_size = len(self.data_loader)
            
        if self.data_loader.batch_size != 1:
            print('WAIT! PASS --batch_size = 1 to do this. Your batch_size is not 1')
            pass
        for step in range(self.evaluation_size):
            if (step+1) % self.evaluation_step == 0:
                [print() for i in range(10)]
                print("[{}/{}] tests".format(step+1, len(self.data_loader)))
                [print() for i in range(10)]
            self.model.eval() 
            self.test_and_error();
    
    def test(self): #receives single image --> can be easily modified to handle multiple images
        'Takes single LR image as input. Returns LR image + (models approx) HR image concatenated'
        'image location must be given by flag --test_image_path'
        self.model.eval()
        step = self.start_step + 1 # if not loading trained start = 0 
        lr_image = Image.open(self.test_image_path)
        lr_image_size = lr_image.size[0]
        #CONSIDER RGB IMAGE
        
        from utils import Kernels, load_kernels
        K, P = load_kernels(file_path='kernels/', scale_factor=2)
        randkern = Kernels(K, P)

        # get LR_RESIDUAL --> [-1,1]
        transform_to_vlr_bicub = transforms.Compose([
                            transforms.Lambda(lambda x: randkern.RandomBlur(x)), #random blur
                            transforms.Lambda(lambda x: bicub_downscale(x,self.scale_factor)),
                            transforms.Resize((lr_image_size, lr_image_size), Image.BICUBIC) #upscale pro tamanho LR
                    ])
        transform_to_vlr_bilin = transforms.Compose([
                            transforms.Lambda(lambda x: randkern.RandomBlur(x)), #random blur
                            transforms.Lambda(lambda x: bilin_downscale(x,self.scale_factor)),
                            transforms.Resize((lr_image_size, lr_image_size), Image.BICUBIC) #upscale pro tamanho LR
                    ])
        transform_to_vlr_nears = transforms.Compose([
                            transforms.Lambda(lambda x: randkern.RandomBlur(x)), #random blur
                            transforms.Lambda(lambda x: nearest_downscale(x,self.scale_factor)),
                            transforms.Resize((lr_image_size, lr_image_size), Image.BICUBIC) #upscale pro tamanho LR
                    ])
        lr_image_hat_bicub = transform_to_vlr_bicub(lr_image)
        lr_residual_bicub = np.array(lr_image).astype(np.float32) - np.array(lr_image_hat_bicub).astype(np.float32)
        lr_residual_bicub_scaled = Scaling(lr_residual_bicub)

        lr_image_hat_bilin = transform_to_vlr_bilin(lr_image)
        lr_residual_bilin = np.array(lr_image).astype(np.float32) - np.array(lr_image_hat_bilin).astype(np.float32)
        lr_residual_bilin_scaled = Scaling(lr_residual_bilin)

        lr_image_hat_nears = transform_to_vlr_nears(lr_image)
        lr_residual_nears = np.array(lr_image).astype(np.float32) - np.array(lr_image_hat_nears).astype(np.float32)
        lr_residual_nears_scaled = Scaling(lr_residual_nears)

        # LR_image_scaled + kernel_proj + LR_residual_scaled (CONCAT) ---> TO TORCH
        lr_image_scaled = Scaling(lr_image)

        # LR_image to torch

        lr_image_without_kernel = lr_image_scaled #self.randkern.ConcatDegraInfo(lr_image_scaled)
        lr_image_with_resid  = np.concatenate((lr_image_without_kernel,\
                                               lr_residual_bicub_scaled,\
                                               lr_residual_bilin_scaled,\
                                               lr_residual_nears_scaled), axis=-1)
        lr_image_with_resid = torch.from_numpy(lr_image_with_resid).float().to(self.device) # NUMPY to TORCH

        # LR_image to torch

        lr_image_scaled = torch.from_numpy(lr_image_scaled).float().to(self.device) # NUMPY to TORCH

        #Transpose - Permute since for model we need input with channels first
        lr_image_scaled = lr_image_scaled.permute(2,0,1) 
        lr_image_with_resid = lr_image_with_resid.permute(2,0,1)

        lr_image_with_resid = lr_image_with_resid.unsqueeze(0) #just add one dimension (index on batch)
        lr_image_scaled = lr_image_scaled.unsqueeze(0)

        lr_image, x = lr_image_scaled, lr_image_with_resid 
        lr_image, x = lr_image.to(self.device), x.to(self.device)


        reconst = self.model(x)

        tmp1 = lr_image.data.cpu().numpy().transpose(0,2,3,1)*255
        image_list = [np.array(Image.fromarray(tmp1.astype(np.uint8)[i]).resize((128,128), Image.BICUBIC)) \
                      for i in range(self.data_loader.batch_size)]
        image_hr_bicubic= np.stack(image_list)
        image_hr_bicubic_single = np.squeeze(image_hr_bicubic)
        #return this ^
        
        image_hr_bicubic = image_hr_bicubic.transpose(0,3,1,2)
        image_hr_bicubic = Scaling(image_hr_bicubic)
        image_hr_bicubic = torch.from_numpy(image_hr_bicubic).float().to(self.device) # NUMPY to TORCH
        hr_image_hat = reconst + image_hr_bicubic
        hr_image_hat_np = hr_image_hat.data.cpu().numpy()
        hr_image_hat_np_scaled = Scaling01(hr_image_hat_np)
        hr_image_hat_np_scaled = np.squeeze(hr_image_hat_np_scaled).transpose((1, 2, 0))
        hr_image_hat_np_png = (hr_image_hat_np_scaled*255).astype(np.uint8)
        #return this ^

        #Saving Image Bicubic and HR Image Hat
        Image.fromarray(image_hr_bicubic_single).save('./results/HR_bicub_images/'+ os.path.basename(self.test_image_path)+'_hr_bic_{}.png'.format(step))
        Image.fromarray(hr_image_hat_np_png).save('./results/HR_HAT_images/'+ os.path.basename(self.test_image_path)+'_hr_hat_{}.png'.format(step))

        #Create Grid
        hr_image_hat_np_scaled = Scaling01(hr_image_hat_np)
        hr_image_hat_torch = torch.from_numpy(hr_image_hat_np_scaled).float().to(self.device) # NUMPY to TORCH

        pairs = torch.cat((image_hr_bicubic.data, \
                        hr_image_hat_torch.data), dim=3)
        grid = make_grid(pairs, 1) 
        tmp = np.squeeze(grid.cpu().numpy().transpose((1, 2, 0)))
        tmp = (255 * tmp).astype(np.uint8)
        random_number = np.random.rand(1)[0]        
        Image.fromarray(tmp).save('./results/grids/'+ os.path.basename(self.test_image_path).split('.')[0]+'_grid_{}.png'.format(step))

        
    def many_tests(self):
        '''Pass just image_folder_path via --test_image_path and it will test for all pictures in the folder'''
        import glob
        TYPES = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        image_paths = []
        root = self.test_image_path
        for ext in TYPES:
            image_paths.extend(glob.glob(os.path.join(root, ext)))
        for img_path in image_paths:
            self.test_image_path = img_path
            self.test()

