import os
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import wandb
wandb.login()

class Trainer(object):
    def __init__(self, model, run_num, device, model_save_path):
        super().__init__()
        self.model = model
        self.run_num = run_num
        self.device = device
        self.model_save_path = model_save_path

    def update_saved_model(self, name):
        """Update saved model (checkpoints and if validation loss is minimized)"""
        if not os.path.isdir(self.model_save_path):
            os.mkdir(self.model_save_path)
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        torch.save(self.model.module.state_dict(), os.path.join(run_path, name + '.pth'))

    def generate_state_dict(self, epoch_num, metrics, optimizer, scheduler=None):
        """Returns a dictionary of the state_dicts of all states but not the model."""
        state = {
            'current_epoch': epoch_num + 1,
            'loss_tracker': metrics,
            'optimizer_state_dict': optimizer.state_dict(), 
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
        }
        return state

    def save_model_and_states_checkpoint(self, epoch_num, metrics, optimizer, scheduler=None):
        """Save a checkpoint state that can be loaded to continue training."""
        state_dict = self.generate_state_dict(epoch_num, metrics, optimizer, scheduler)
        state_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        self.update_saved_model('checkpoint_model')
        torch.save(state_dict, os.path.join(state_path, 'checkpoint.state'))

    def load_state_checkpoint(self, optimizer, scheduler=None):
        """Load everything but the model."""
        checkpoint_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        checkpoint_fname = os.path.join(checkpoint_path, 'checkpoint.state')
        try: 
            os.path.exists(checkpoint_fname)
        except: 
            raise FileNotFoundError(f"Checkpoint not found in {checkpoint_fname}")
        state_dict = torch.load(checkpoint_fname)
        current_epoch = state_dict['current_epoch']
        metrics = state_dict['loss_tracker']
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        if state_dict['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        return current_epoch, metrics, optimizer, scheduler

    def generate_plot(self, in_dp, out_dp, gt_amp, pred_amp, gt_ph, pred_ph, filename):
        f, ax = plt.subplots(nrows=2, ncols=3)

        in0 = ax[0, 0].imshow(in_dp, interpolation='none', norm=colors.LogNorm(), cmap='jet')
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(in0, cax=cax, orientation='vertical')
        ax[0, 0].set_title('Input Diff. Amp.')

        in1 = ax[0, 1].imshow(gt_amp, interpolation='none')
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(in1, cax=cax, orientation='vertical', format='%.1f')
        ax[0, 1].set_title('GT Amp.')

        in2 = ax[0, 2].imshow(gt_ph, interpolation='none', cmap='magma')
        divider = make_axes_locatable(ax[0, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(in2, cax=cax, orientation='vertical', format='%.1f')
        ax[0, 2].set_title('GT Phase')

        out0 = ax[1, 0].imshow(out_dp, interpolation='none', norm=colors.LogNorm(), cmap='jet')
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(out0, cax=cax, orientation='vertical')
        ax[1, 0].set_title('Output Diff. Amp.')

        out1 = ax[1, 1].imshow(pred_amp, interpolation='none')
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(out1, cax=cax, orientation='vertical', format='%.1f')
        ax[1, 1].set_title('Predicted Amp.')

        out2 = ax[1, 2].imshow(pred_ph, interpolation='none', cmap='magma')
        divider = make_axes_locatable(ax[1, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(out2, cax=cax, orientation='vertical', format='%.1f')
        ax[1, 2].set_title('Predicted Phase')

        plt.tight_layout()
        f.savefig(os.path.join(self.model_save_path, filename), bbox_inches='tight', transparent=True)

    def train(self, trainloader, criterion, optimizer, metrics):
        """Training loop"""
        running_loss = 0.0
        running_amp_loss = 0.0
        running_ph_loss = 0.0

        for i, (diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale) in enumerate(trainloader):
            input_diff = diff_amp.to(self.device)
            input_probe = torch.view_as_real(probe.clone().detach()).to(self.device)
            input_norm = norm.to(self.device)
            input_scale = scale.to(self.device)

            output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)

            loss = criterion(output_diff, input_diff)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()

            # Also track the amplitude and phase loss to see if the network is predicting something reasonable
            loss_amp = criterion(output_amp.detach().cpu(), amp_patch)
            loss_ph = criterion(output_ph.detach().cpu(), ph_patch)  
            running_amp_loss += loss_amp
            running_ph_loss += loss_ph

        wandb.log({"train_loss": running_loss/i})
        wandb.log({"train_amp_loss": running_amp_loss/i})
        wandb.log({"train_ph_loss": running_ph_loss/i})
        metrics['training_loss'].append(running_loss/i)
        metrics['train_amp_loss'].append(running_amp_loss/i)
        metrics['train_ph_loss'].append(running_ph_loss/i)

    def validate(self, validloader, criterion, optimizer, metrics, plot=False, scheduler=None):
        """Validation loop"""
        val_loss = 0.0
        val_amp_loss = 0.0
        val_ph_loss = 0.0
        plot_counter = 0

        for j, (diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale) in enumerate(validloader):
            input_diff = diff_amp.to(self.device)
            input_probe = torch.view_as_real(probe.clone().detach()).to(self.device)
            input_norm = norm.to(self.device)
            input_scale = scale.to(self.device)

            output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)

            loss = criterion(output_diff, input_diff)
            val_loss += loss.detach().item()

            loss_amp = criterion(output_amp.detach().cpu(), amp_patch)
            loss_ph = criterion(output_ph.detach().cpu(), ph_patch)
            val_amp_loss += loss_amp
            val_ph_loss += loss_ph

        wandb.log({"val_loss": val_loss/j})
        wandb.log({"val_amp_loss": val_amp_loss/j})
        wandb.log({"val_ph_loss": val_ph_loss/j})
        metrics['validation_loss'].append(val_loss/j)
        metrics['val_amp_loss'].append(val_amp_loss/j)
        metrics['val_ph_loss'].append(val_ph_loss/j)

        if plot:
            input_diff = input_diff.squeeze().detach().cpu().numpy()[0]
            output_diff = output_diff.squeeze().detach().cpu().numpy()[0]
            input_amp = amp_patch[0, 0]
            output_amp = output_amp.squeeze().detach().cpu().numpy()[0]
            input_ph = ph_patch[0, 0]
            output_ph = output_ph.squeeze().detach().cpu().numpy()[0]
            filename = 'plot' + str(plot_counter) + '.png'
            self.generate_plot(input_diff, output_diff, input_amp, output_amp, input_ph, output_ph, filename)
            wandb.log({"val_plot": wandb.Image(os.path.join(self.model_save_path, filename), caption="Training progress")})
            plot_counter += 1

        if scheduler:
            scheduler.step(val_loss/j)
            metrics['lr'].append(optimizer.param_groups[0]['lr'])
            wandb.log({"lr": optimizer.param_groups[0]['lr']})

        if (val_loss/j < metrics['best_val_loss']):
            print("Saving improved model after Val. Loss improved from %.4f to %.5f" 
                  % (metrics['best_val_loss'], val_loss/j))
            metrics['best_val_loss'] = val_loss/j
            self.update_saved_model('best_model')