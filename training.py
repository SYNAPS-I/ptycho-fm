import os
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from utils.ptychi_utils import place_patches_fourier_shift

import wandb


class Trainer(object):
    def __init__(self, model, mode, run_num, device, model_save_path,
                 is_main_process=True, use_ddp=False, wandb_enabled=True):
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

        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            torch.save(self.model.module.state_dict(), os.path.join(run_path, name + '.pth'))
        else:
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
        except Exception:
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

    def generate_test_plot(self, dataloader, epoch, filename):
        total_scan_points = len(dataloader.dataset)
        pred_amp = torch.zeros((total_scan_points, dataloader.dataset.pattern_shape[0], dataloader.dataset.pattern_shape[1]), device='cpu')
        pred_ph = torch.zeros(pred_amp.shape, device='cpu')
        gt_amp = torch.zeros(pred_amp.shape, device='cpu')
        gt_ph = torch.zeros(pred_amp.shape, device='cpu')
        scan_idx = 0

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch
                batch_size = diff_amp.size(0)

                input_diff = diff_amp.to(self.device, non_blocking=True)
                input_probe = torch.view_as_real(probe.clone().detach()).to(self.device, non_blocking=True)
                input_norm = norm.to(self.device, non_blocking=True)
                input_scale = scale.to(self.device, non_blocking=True)

                _output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)
                pred_amp[scan_idx:scan_idx + batch_size] = output_amp.squeeze().detach().cpu()
                pred_ph[scan_idx:scan_idx + batch_size] = output_ph.squeeze().detach().cpu()
                gt_amp[scan_idx:scan_idx + batch_size] = amp_patch.squeeze().detach().cpu()
                gt_ph[scan_idx:scan_idx + batch_size] = ph_patch.squeeze().detach().cpu()

                scan_idx += batch_size

        if dataloader.dataset._cached_probe_positions is None:
            dataloader.dataset._cache_positions()
            dataloader.dataset._cache_object_data()

        object_size = dataloader.dataset.object_shape
        positions = dataloader.dataset._cached_probe_positions
        pred_amp_object = torch.zeros(object_size, device='cpu')
        pred_ph_object = torch.zeros(object_size, device='cpu')
        buffer = torch.zeros(object_size, device='cpu')

        pred_amp_object = place_patches_fourier_shift(
            pred_amp_object,
            positions,
            pred_amp[:, 64:-64, 64:-64],
            op="add",
            adjoint_mode=False,
            pad=32
        )
        pred_ph_object = place_patches_fourier_shift(
            pred_ph_object,
            positions,
            pred_ph[:, 64:-64, 64:-64],
            op="add",
            adjoint_mode=False,
            pad=32
        )
        buffer = place_patches_fourier_shift(
            buffer,
            positions,
            torch.ones_like(pred_ph[:, 64:-64, 64:-64]),
            op="add",
            adjoint_mode=False,
            pad=32
        )

        gt_amp_object = torch.zeros(object_size, device='cpu')
        gt_ph_object = torch.zeros(object_size, device='cpu')
        gt_amp_object = place_patches_fourier_shift(
            gt_amp_object,
            positions,
            gt_amp[:, 64:-64, 64:-64],
            op="add",
            adjoint_mode=False,
            pad=32
        )
        gt_ph_object = place_patches_fourier_shift(
            gt_ph_object,
            positions,
            gt_ph[:, 64:-64, 64:-64],
            op="add",
            adjoint_mode=False,
            pad=32
        )

        pred_amp_object = pred_amp_object / torch.clip(buffer, min=1)
        pred_ph_object = pred_ph_object / torch.clip(buffer, min=1)
        gt_amp_object = gt_amp_object / torch.clip(buffer, min=1)
        gt_ph_object = gt_ph_object / torch.clip(buffer, min=1)

        vmin_ph = torch.mean(pred_ph_object[180:-180, 180:-180]) - (2 * torch.std(pred_ph_object[180:-180, 180:-180]))
        vmax_ph = torch.mean(pred_ph_object[180:-180, 180:-180]) + (2 * torch.std(pred_ph_object[180:-180, 180:-180]))

        f, ax = plt.subplots(figsize=(9, 8), ncols=2, nrows=2)

        gt0 = ax[0, 0].imshow(gt_amp_object[180:-180, 180:-180], interpolation='none', vmin=0.9, vmax=1)
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(gt0, cax=cax, orientation='vertical')
        ax[0, 0].set_title('LSQML amplitude')

        gt1 = ax[0, 1].imshow(gt_ph_object[180:-180, 180:-180], interpolation='none', vmin=-1.3, vmax=1.3, cmap='magma')
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(gt1, cax=cax, orientation='vertical')
        ax[0, 1].set_title('LSQML phase')

        pred0 = ax[1, 0].imshow(pred_amp_object[180:-180, 180:-180], interpolation='none', vmin=0.9, vmax=1)
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(pred0, cax=cax, orientation='vertical')
        ax[1, 0].set_title('Predicted amplitude')

        pred1 = ax[1, 1].imshow(pred_ph_object[180:-180, 180:-180], interpolation='none', vmin=vmin_ph, vmax=vmax_ph, cmap='magma')
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(pred1, cax=cax, orientation='vertical')
        ax[1, 1].set_title('Predicted phase')

        plt.tight_layout()
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        f.savefig(os.path.join(run_path, filename), bbox_inches='tight', transparent=True)
        plt.close(f)

        if self.wandb_enabled:
            wandb.log({"test_plot": wandb.Image(os.path.join(run_path, filename), caption=f"Test: epoch {epoch}")})

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

        total_batches = len(dataloader)
        progress_milestones = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        milestone_batches = [int(total_batches * m) for m in progress_milestones]
        next_milestone_idx = 0

        import time
        from datetime import datetime

        if self.is_main_process:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training loop: {total_batches} total batches", flush=True)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading first batch... (this may take a while with lazy data loading)", flush=True)

        batch_0_io_start = time.time()
        train_end_time = None

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx < 10:
                if batch_idx == 0:
                    io_time = time.time() - batch_0_io_start
                else:
                    io_time = time.time() - train_end_time

            if batch_idx == 0 and self.is_main_process:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] First batch loaded! Starting training... (batch 1/{total_batches})", flush=True)
            elif batch_idx < 10 and self.is_main_process:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing batch {batch_idx + 1}/{total_batches}", flush=True)

            # ── Unpack ──────────────────────────────────────────────────────────
            diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch

            # ── FIX: move required tensors to device right after unpack ─────────
            # Works for both normal DataLoader (CPU) and CUDAPrefetcher (already GPU)
            diff_amp  = diff_amp.to(self.device, non_blocking=True)
            amp_patch = amp_patch.to(self.device, non_blocking=True)
            ph_patch  = ph_patch.to(self.device, non_blocking=True)
            norm      = norm.to(self.device, non_blocking=True)
            scale     = scale.to(self.device, non_blocking=True)

            if batch_idx < 10:
                train_start_time = time.time()

            input_diff = diff_amp
            input_probe = torch.view_as_real(probe.clone().detach()).to(self.device, non_blocking=True)
            input_norm = norm
            input_scale = scale

            output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)

            if self.mode == 'supervised':
                # amp_patch/ph_patch already on device
                loss = criterion(output_amp, amp_patch) + criterion(output_ph, ph_patch)
            else:
                loss = criterion(output_diff, input_diff)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_end_time = time.time()

            if batch_idx < 10:
                train_time = train_end_time - train_start_time
                total_time = io_time + train_time
                if self.is_main_process:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Batch {batch_idx + 1} Timing] IO: {io_time:.3f}s | Training: {train_time:.3f}s | Total: {total_time:.3f}s", flush=True)

            running_loss += loss.detach().item()

            # Track amp/phase losses (targets now guaranteed on same device)
            loss_amp = criterion(output_amp.detach(), amp_patch)
            loss_ph  = criterion(output_ph.detach(), ph_patch)
            running_amp_loss += loss_amp.item()
            running_ph_loss  += loss_ph.item()

            if batch_idx > 0 and self.is_main_process:
                from datetime import datetime
                if batch_idx < 1000 and (batch_idx + 1) % 100 == 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Training] Batch {batch_idx + 1}/{total_batches} ({(batch_idx + 1) * 100.0 / total_batches:.2f}%)", flush=True)
                elif (batch_idx + 1) % 1000 == 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Training] Batch {batch_idx + 1}/{total_batches} ({(batch_idx + 1) * 100.0 / total_batches:.2f}%)", flush=True)

            if next_milestone_idx < len(milestone_batches) and batch_idx >= milestone_batches[next_milestone_idx]:
                from datetime import datetime
                progress_pct = int(progress_milestones[next_milestone_idx] * 100)
                if self.is_main_process:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Training Progress] {progress_pct}% complete ({batch_idx + 1}/{total_batches} batches)", flush=True)
                next_milestone_idx += 1

        num_batches = len(dataloader)
        avg_train_loss = running_loss / num_batches
        avg_amp_loss = running_amp_loss / num_batches
        avg_ph_loss = running_ph_loss / num_batches

        if self.is_main_process:
            from datetime import datetime
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training epoch complete: {num_batches} batches processed", flush=True)

        avg_train_loss = self.synchronize_loss(avg_train_loss)
        avg_amp_loss = self.synchronize_loss(avg_amp_loss)
        avg_ph_loss = self.synchronize_loss(avg_ph_loss)

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
        """
        if self.is_main_process:
            from datetime import datetime
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting validation loop: {len(dataloader)} batches", flush=True)

        val_loss = 0.0
        val_amp_loss = 0.0
        val_ph_loss = 0.0

        total_batches = len(dataloader)
        progress_milestones = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        milestone_batches = [int(m * total_batches) for m in progress_milestones]
        next_milestone_idx = 0

        last_input_diff = None
        last_output_diff = None
        last_amp_patch = None
        last_output_amp = None
        last_ph_patch = None
        last_output_ph = None

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch

                # ── FIX: move required tensors to device right after unpack ─────────
                diff_amp  = diff_amp.to(self.device, non_blocking=True)
                amp_patch = amp_patch.to(self.device, non_blocking=True)
                ph_patch  = ph_patch.to(self.device, non_blocking=True)
                norm      = norm.to(self.device, non_blocking=True)
                scale     = scale.to(self.device, non_blocking=True)

                input_diff = diff_amp
                input_probe = torch.view_as_real(probe.clone().detach()).to(self.device, non_blocking=True)
                input_norm = norm
                input_scale = scale

                output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)

                if self.mode == 'supervised':
                    loss = criterion(output_amp, amp_patch) + criterion(output_ph, ph_patch)
                else:
                    loss = criterion(output_diff, input_diff)

                val_loss += loss.detach().item()

                loss_amp = criterion(output_amp.detach(), amp_patch)
                loss_ph  = criterion(output_ph.detach(), ph_patch)
                val_amp_loss += loss_amp.item()
                val_ph_loss += loss_ph.item()

                if next_milestone_idx < len(milestone_batches) and batch_idx >= milestone_batches[next_milestone_idx]:
                    from datetime import datetime
                    progress_pct = int(progress_milestones[next_milestone_idx] * 100)
                    if self.is_main_process:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Validation Progress] {progress_pct}% complete ({batch_idx + 1}/{total_batches} batches)", flush=True)
                    next_milestone_idx += 1

                if plot and self.is_main_process:
                    last_input_diff = input_diff
                    last_output_diff = output_diff
                    last_amp_patch = amp_patch
                    last_output_amp = output_amp
                    last_ph_patch = ph_patch
                    last_output_ph = output_ph

        num_batches = len(dataloader)
        avg_val_loss = val_loss / num_batches
        avg_val_amp_loss = val_amp_loss / num_batches
        avg_val_ph_loss = val_ph_loss / num_batches

        if self.is_main_process:
            from datetime import datetime
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation epoch complete: {num_batches} batches processed", flush=True)

        avg_val_loss = self.synchronize_loss(avg_val_loss)
        avg_val_amp_loss = self.synchronize_loss(avg_val_amp_loss)
        avg_val_ph_loss = self.synchronize_loss(avg_val_ph_loss)

        if self.is_main_process and self.wandb_enabled:
            wandb.log({"val_loss": avg_val_loss})
            wandb.log({"val_amp_loss": avg_val_amp_loss})
            wandb.log({"val_ph_loss": avg_val_ph_loss})

        metrics['validation_loss'].append(avg_val_loss)
        metrics['val_amp_loss'].append(avg_val_amp_loss)
        metrics['val_ph_loss'].append(avg_val_ph_loss)

        if plot and self.is_main_process and last_input_diff is not None:
            input_diff_np = last_input_diff[0, 0].detach().cpu().numpy()
            output_diff_np = last_output_diff[0, 0].detach().cpu().numpy()

            input_amp = last_amp_patch[0, 0].detach().cpu()
            output_amp = last_output_amp[0, 0].detach().cpu().numpy()

            input_ph = last_ph_patch[0, 0].detach().cpu()
            output_ph = last_output_ph[0, 0].detach().cpu().numpy()

            filename = 'plot_epoch' + str(epoch) + '.png'
            self.generate_plot(input_diff_np, output_diff_np, input_amp, output_amp, input_ph, output_ph, filename)

            if self.wandb_enabled:
                run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
                wandb.log({"val_plot": wandb.Image(os.path.join(run_path, filename), caption=f"Epoch {epoch}")})

        if scheduler:
            scheduler.step(avg_val_loss)
            metrics['lr'].append(optimizer.param_groups[0]['lr'])
            if self.is_main_process and self.wandb_enabled:
                wandb.log({"lr": optimizer.param_groups[0]['lr']})

        if avg_val_loss < metrics['best_val_loss']:
            if self.is_main_process:
                print("Saving improved model after Val. Loss improved from %.4f to %.5f"
                      % (metrics['best_val_loss'], avg_val_loss), flush=True)
                self.update_saved_model('best_model')
            metrics['best_val_loss'] = avg_val_loss
