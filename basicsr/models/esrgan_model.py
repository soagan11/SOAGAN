import importlib
import torch
from collections import OrderedDict

from basicsr.models.srgan_model import SRGANModel

loss_module = importlib.import_module('basicsr.models.losses')


class ESRGANModel(SRGANModel):
    """modification ESRGAN model . Unet鉴别器损失函数"""

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0
                and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(
                    self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss (relativistic gan)
            # print(self.lq.shape, self.output.shape, self.gt.shape, '\n')
            real_d_pred, real_d_out = self.net_d(self.gt)[0].detach(), self.net_d(self.gt)[1].detach()
            fake_g_pred, fake_g_out = self.net_d(self.output)
            l_g_real = self.cri_gan(
                real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            # print(real_d_pred.shape, fake_g_pred.shape, real_d_out.shape, fake_g_out.shape, '\n')
            l_g_real = l_g_real + self.cri_gan(
                (real_d_out - fake_g_out), False, is_disc=False)

            l_g_fake = self.cri_gan(
                fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)

            l_g_fake = l_g_fake + self.cri_gan(
                (fake_g_out - real_d_out), True, is_disc=False)

            l_g_gan = (l_g_real + l_g_fake) / 4

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # gan loss (relativistic gan)

        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.

        # real
        fake_d_pred, fake_d_output = self.net_d(self.output)[0].detach(), self.net_d(self.output)[1].detach()
        real_d_pred, real_d_out = self.net_d(self.gt)
        # out1
        l_d_real = self.cri_gan(
            real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        # l_d_real.backward()
        # out2
        l_d_real = l_d_real + self.cri_gan(
            (real_d_out - fake_d_output),
            True,
            is_disc=True) * 0.5
        l_d_real.backward()
        # fake
        fake_d_pred, fake_d_output = self.net_d(self.output.detach())
        real_d_pred, real_d_out = self.net_d(self.gt)
        # out1
        l_d_fake = self.cri_gan(
            fake_d_pred - torch.mean(real_d_pred.detach()),
            False,
            is_disc=True) * 0.5
        # out2
        l_d_fake = l_d_fake + self.cri_gan(
            (fake_d_output - real_d_out.detach()),
            False,
            is_disc=True) * 0.5
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)
