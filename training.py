import os
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import wandb

class Trainer(object):
    def __init__(self, model, mode, run_num, device, model_save_path, is_main_process=True, use_ddp=False, wandb_enabled=True):
        super().__init__()
        self.model = model
        self.mode = mode
        self.run_num = run_num
        self.device = device
        self.model_save_path = model_save_path
        self.is_main_process = is_main_process
        self.use_ddp = use_ddp
        self.wandb_enabled = wandb_enabled

    def synchronize_loss(self, loss_value):
        """Synchronize loss across all processes in DDP."""
        if self.use_ddp and dist.is_initialized():
            loss_tensor = torch.tensor(loss_value, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            # Average across all processes
            loss_tensor /= dist.get_world_size()
            return loss_tensor.item()
        return loss_value

    def update_saved_model(self, name):
        """Update saved model (checkpoints and if validation loss is minimized)"""
        if not os.path.isdir(self.model_save_path):
            os.mkdir(self.model_save_path)
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        # Handle both DataParallel and DistributedDataParallel
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            # Save the original model's state_dict
            torch.save(self.model.module.state_dict(), os.path.join(run_path, name + '.pth'))
        else:
            # Save the state_dict for a non-parallel model
            torch.save(self.model.state_dict(), os.path.join(run_path, name + '.pth'))

    def save_config(self, config_path='config.yaml'):
        """Save a copy of the config file to the run path for reproducibility."""
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        if not os.path.isdir(run_path):
            os.makedirs(run_path, exist_ok=True)
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(run_path, 'config.yaml'))

    def generate_state_dict(self, epoch_num, metrics, optimizer, wandb_run_id=None, scheduler=None):
        """Returns a dictionary of the state_dicts of all states but not the model."""
        state = {
            'current_epoch': epoch_num + 1,
            'loss_tracker': metrics,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'wandb_run_id': wandb_run_id
        }
        return state

    def save_model_and_states_checkpoint(self, epoch_num, metrics, optimizer=None, wandb_run_id=None, scheduler=None):
        """Save a checkpoint state that can be loaded to continue training."""
        state_dict = self.generate_state_dict(epoch_num, metrics, optimizer, wandb_run_id, scheduler)
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
        wandb_run_id = state_dict.get('wandb_run_id', None)
        return current_epoch, metrics, optimizer, wandb_run_id, scheduler

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
        f.colorbar(in1, cax=cax, orientation='vertical', format='%.2f')
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
        f.colorbar(out1, cax=cax, orientation='vertical', format='%.2f')
        ax[1, 1].set_title('Predicted Amp.')

        out2 = ax[1, 2].imshow(pred_ph, interpolation='none', cmap='magma')
        divider = make_axes_locatable(ax[1, 2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(out2, cax=cax, orientation='vertical', format='%.1f')
        ax[1, 2].set_title('Predicted Phase')

        plt.tight_layout()
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        f.savefig(os.path.join(run_path, filename), bbox_inches='tight', transparent=True)

    def train(self, dataloader, criterion, optimizer, metrics):
        """
        Training loop.

        Args:
            dataloader: PyTorch DataLoader yielding batches of
                       (diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale)
        """
        running_loss = 0.0
        running_amp_loss = 0.0
        running_ph_loss = 0.0

        for batch in dataloader:
            # Unpack batch
            diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch

            input_diff = diff_amp.to(self.device)
            input_probe = torch.view_as_real(probe.clone().detach()).to(self.device)
            input_norm = norm.to(self.device)
            input_scale = scale.to(self.device)

            output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)

            if self.mode == 'supervised':
                loss = criterion(output_amp, amp_patch.to(self.device)) + criterion(output_ph, ph_patch.to(self.device))
            else:
                loss = criterion(output_diff, input_diff)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()

            # Also track the amplitude and phase loss to see if the network is predicting something reasonable
            loss_amp = criterion(output_amp.detach().cpu(), amp_patch)
            loss_ph = criterion(output_ph.detach().cpu(), ph_patch)
            running_amp_loss += loss_amp.item()
            running_ph_loss += loss_ph.item()

        # Calculate average losses (use len(dataloader) for batch count)
        num_batches = len(dataloader)
        avg_train_loss = running_loss / num_batches
        avg_amp_loss = running_amp_loss / num_batches
        avg_ph_loss = running_ph_loss / num_batches

        # Synchronize losses across all ranks for DDP
        avg_train_loss = self.synchronize_loss(avg_train_loss)
        avg_amp_loss = self.synchronize_loss(avg_amp_loss)
        avg_ph_loss = self.synchronize_loss(avg_ph_loss)

        # Log and save metrics (synchronized values)
        if self.is_main_process and self.wandb_enabled:
            wandb.log({"train_loss": avg_train_loss})
            wandb.log({"train_amp_loss": avg_amp_loss})
            wandb.log({"train_ph_loss": avg_ph_loss})

        metrics['training_loss'].append(avg_train_loss)
        metrics['train_amp_loss'].append(avg_amp_loss)
        metrics['train_ph_loss'].append(avg_ph_loss)

    def validate(self, dataloader, criterion, optimizer, metrics, plot=False, epoch=0, scheduler=None):
        """
        Validation loop.

        Args:
            dataloader: PyTorch DataLoader yielding batches of
                       (diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale)
            epoch: Current epoch number for plot naming
        """
        val_loss = 0.0
        val_amp_loss = 0.0
        val_ph_loss = 0.0

        # Variables for plotting (save last batch)
        last_input_diff = None
        last_output_diff = None
        last_amp_patch = None
        last_output_amp = None
        last_ph_patch = None
        last_output_ph = None

        # Use no_grad() to prevent gradient computation during validation
        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch
                diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch

                input_diff = diff_amp.to(self.device)
                input_probe = torch.view_as_real(probe.clone().detach()).to(self.device)
                input_norm = norm.to(self.device)
                input_scale = scale.to(self.device)

                output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)

                if self.mode == 'supervised':
                    loss = criterion(output_amp, amp_patch.to(self.device)) + criterion(output_ph, ph_patch.to(self.device))
                else:
                    loss = criterion(output_diff, input_diff)
                    #loss = criterion(torch.log10(output_diff + 1.0e-6), torch.log10(input_diff + 1.0e-6))
                val_loss += loss.detach().item()

                loss_amp = criterion(output_amp.detach().cpu(), amp_patch)
                loss_ph = criterion(output_ph.detach().cpu(), ph_patch)
                val_amp_loss += loss_amp.item()
                val_ph_loss += loss_ph.item()

                # Save last batch for plotting
                if plot and self.is_main_process:
                    last_input_diff = input_diff
                    last_output_diff = output_diff
                    last_amp_patch = amp_patch
                    last_output_amp = output_amp
                    last_ph_patch = ph_patch
                    last_output_ph = output_ph

        # Calculate average losses (use len(dataloader) for batch count)
        num_batches = len(dataloader)
        avg_val_loss = val_loss / num_batches
        avg_val_amp_loss = val_amp_loss / num_batches
        avg_val_ph_loss = val_ph_loss / num_batches

        # Synchronize losses across all ranks for DDP
        avg_val_loss = self.synchronize_loss(avg_val_loss)
        avg_val_amp_loss = self.synchronize_loss(avg_val_amp_loss)
        avg_val_ph_loss = self.synchronize_loss(avg_val_ph_loss)

        # Log and save metrics (synchronized values)
        if self.is_main_process and self.wandb_enabled:
            wandb.log({"val_loss": avg_val_loss})
            wandb.log({"val_amp_loss": avg_val_amp_loss})
            wandb.log({"val_ph_loss": avg_val_ph_loss})

        metrics['validation_loss'].append(avg_val_loss)
        metrics['val_amp_loss'].append(avg_val_amp_loss)
        metrics['val_ph_loss'].append(avg_val_ph_loss)

        if plot and self.is_main_process and last_input_diff is not None:
            input_diff = last_input_diff.squeeze().detach().cpu().numpy()[0]
            output_diff = last_output_diff.squeeze().detach().cpu().numpy()[0]
            input_amp = last_amp_patch[0, 0]
            output_amp = last_output_amp.squeeze().detach().cpu().numpy()[0]
            input_ph = last_ph_patch[0, 0]
            output_ph = last_output_ph.squeeze().detach().cpu().numpy()[0]
            filename = 'plot_epoch' + str(epoch) + '.png'
            self.generate_plot(input_diff, output_diff, input_amp, output_amp, input_ph, output_ph, filename)
            if self.wandb_enabled:
                run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
                wandb.log({"val_plot": wandb.Image(os.path.join(run_path, filename), caption=f"Epoch {epoch}")})

        if scheduler:
            scheduler.step(avg_val_loss)
            metrics['lr'].append(optimizer.param_groups[0]['lr'])
            if self.is_main_process and self.wandb_enabled:
                wandb.log({"lr": optimizer.param_groups[0]['lr']})

        # Check if this is the best model (use synchronized validation loss)
        # Only save on main process to avoid multiple saves
        if avg_val_loss < metrics['best_val_loss']:
            if self.is_main_process:
                print("Saving improved model after Val. Loss improved from %.4f to %.5f"
                      % (metrics['best_val_loss'], avg_val_loss), flush=True)
                self.update_saved_model('best_model')
            # Update best_val_loss on all ranks so they stay in sync
            metrics['best_val_loss'] = avg_val_loss