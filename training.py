import os
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from torch.profiler import profile, ProfilerActivity, record_function
from utils.ptychi_utils import place_patches_fourier_shift

import wandb
profiling = False #True #False
activities = [ProfilerActivity.CPU]
if torch.cuda.is_available():
    device = "cuda"
    activities += [ProfilerActivity.CUDA]
elif torch.xpu.is_available():
    device = "xpu"
    activities += [ProfilerActivity.XPU]
else:
    print("Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices")
    import sys
    sys.exit(0)
sort_by_keyword = device + "_time_total"
class Trainer(object):
    def __init__(self, model, mode, run_num, device, model_save_path, is_main_process=True, use_ddp=False, use_prefetch=False, wandb_enabled=True):
        super().__init__()
        self.model = model
        self.mode = mode
        self.run_num = run_num
        self.device = device
        self.model_save_path = model_save_path
        self.is_main_process = is_main_process
        self.use_ddp = use_ddp
        self.wandb_enabled = wandb_enabled
        self.use_prefetch = use_prefetch

    def synchronize_loss(self, loss_value):
        """Synchronize loss across all processes in DDP."""
        if self.use_ddp and dist.is_initialized():
            loss_tensor = torch.tensor(loss_value, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            # Average across all processes
            loss_tensor /= dist.get_world_size()
            #print(f"sync'd from all reduce op : {loss_tensor.item()}")
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
        # Note: don't use this with very large datasets, use the inference script for those instead
        pred_amp = torch.zeros((total_scan_points, dataloader.dataset.pattern_shape[0], dataloader.dataset.pattern_shape[1]), device='cpu')
        pred_ph = torch.zeros(pred_amp.shape, device='cpu')
        gt_amp = torch.zeros(pred_amp.shape, device='cpu')
        gt_ph = torch.zeros(pred_amp.shape, device='cpu')
        scan_idx = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch
                batch_size = diff_amp.size(0)

                input_diff = diff_amp.to(self.device)
                input_probe = torch.view_as_real(probe.clone().detach()).to(self.device)
                input_norm = norm.to(self.device)
                input_scale = scale.to(self.device)

                _output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)
                pred_amp[scan_idx:scan_idx+batch_size] = output_amp.squeeze().detach().cpu()
                pred_ph[scan_idx:scan_idx+batch_size] = output_ph.squeeze().detach().cpu()
                gt_amp[scan_idx:scan_idx+batch_size] = amp_patch.squeeze().detach().cpu()
                gt_ph[scan_idx:scan_idx+batch_size] = ph_patch.squeeze().detach().cpu()

                scan_idx += batch_size

        # Stitch the patches together into an object
        # Ensure cache is populated (in case no batches were processed)
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
            pred_amp[:, 64:-64, 64:-64], # Slice only the center 128 x 128 portion of each patch
            op="add", 
            adjoint_mode=False,
            pad=32 # Crop 32 pixels from each edge to remove ripple artifacts from Fourier shifting
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
        # Normalize by the occupancy
        pred_amp_object = pred_amp_object / torch.clip(buffer, min=1)
        pred_ph_object = pred_ph_object / torch.clip(buffer, min=1)
        gt_amp_object = gt_amp_object / torch.clip(buffer, min=1)
        gt_ph_object = gt_ph_object / torch.clip(buffer, min=1)
        
        # Make the plot
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

        # Log to wandb
        if self.wandb_enabled:
            wandb.log({"test_plot": wandb.Image(os.path.join(run_path, filename), caption=f"Test: epoch {epoch}")})

    def trace_handler(self,p):
        if self.is_main_process:
            print("tracing the profiler")
            p.export_chrome_trace("./trace_" + str(p.step_num) + ".json")

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

        # Start timing for first batch (before DataLoader fetch)
        batch_0_io_start = time.perf_counter()
        timing_window_start = 5
        timing_window_end = 15
        io_time_sum = 0.0
        data_move_time_sum = 0.0
        fwd_pass_time_sum = 0.0
        loss_time_sum = 0.0
        bwd_pass_time_sum = 0.0
        opt_step_time_sum = 0.0
        train_time_sum = 0.0
        other_time_sum = 0.0
        timing_samples = 0
        train_end_time = None  # Will be set after each batch completes
        first_print = 0

        def _sync_device():
            if 'cuda' in str(self.device):
                torch.cuda.synchronize(self.device)
            elif 'xpu' in str(self.device):
                torch.xpu.synchronize(self.device)
        if profiling:
            prof = torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU ],
            schedule=torch.profiler.schedule(wait=2, warmup=3, active=2, repeat=2),
            on_trace_ready=self.trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            )
            prof.start()        
        for batch_idx, batch in enumerate(dataloader):
            # Time IO (data loading) - this captures the time to get batch from DataLoader
            # The DataLoader fetch happens at the 'for' line above, so we time from end of previous batch
            if batch_idx == 0:
                # For first batch, we already started timing before the loop
                io_time = time.perf_counter() - batch_0_io_start
            else:
                # For subsequent batches, time from end of previous batch to now
                # This includes the DataLoader fetch time (the 2-4 second gap you're seeing)
                io_time = time.perf_counter() - train_end_time
            #else:
            #    break
            # Print when first batch is loaded and for first 10 batches
            if batch_idx == 0 and self.is_main_process:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] First batch loaded! Starting training... (batch 1/{total_batches})", flush=True)
            elif batch_idx < 10 and self.is_main_process:
                # Print first 10 batches with timestamps to confirm training is working
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing batch {batch_idx + 1}/{total_batches}", flush=True)
           
            _sync_device()
            move_data = time.perf_counter()
            # Unpack batch (this is fast - data is already loaded from DataLoader)
            diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch
            
            input_diff = diff_amp.to(self.device)
            input_probe = torch.view_as_real(probe.clone().detach()).to(self.device)
            input_norm = norm.to(self.device)
            input_scale = scale.to(self.device)
            if self.use_prefetch == False:
                ph_patch = ph_patch.to(self.device)
                amp_patch = amp_patch.to(self.device)
            _sync_device()
            fwd_pass_st = time.perf_counter()
            if profiling:
                with torch.profiler.record_function("model_fwdpass"):
                    output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)
            else:
                output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)
            _sync_device()
            fwd_pass_end = time.perf_counter()

            _sync_device()
            loss_st = time.perf_counter()
            if profiling:
                with torch.profiler.record_function("compute_loss"):
                    if self.mode == 'supervised':
                        loss = criterion(output_amp, amp_patch.to(self.device)) + criterion(output_ph, ph_patch.to(self.device))
                    else:
                        loss = criterion(output_diff, input_diff)
            else:
                if self.mode == 'supervised':
                    loss = criterion(output_amp, amp_patch.to(self.device)) + criterion(output_ph, ph_patch.to(self.device))
                else:
                    loss = criterion(output_diff, input_diff)
            _sync_device()
            loss_end = time.perf_counter()

            _sync_device()
            bwd_pass_st = time.perf_counter()
            if profiling:
                with torch.profiler.record_function("model_backward"):
                    optimizer.zero_grad()
                    loss.backward()
            else:
                optimizer.zero_grad()
                loss.backward()
            _sync_device()
            bwd_pass_end = time.perf_counter()

            _sync_device()
            opt_step_st = time.perf_counter()
            if profiling:
                with torch.profiler.record_function("optimizer_step"):
                    optimizer.step()
            else:
                optimizer.step()
            _sync_device()
            opt_step_end = time.perf_counter()

            if profiling:
                prof.step()
            
            # Time training completion - always track end time for IO timing of next batch
            train_end_time = time.perf_counter()
            if batch_idx == 10:
                if 'cuda' in str(self.device):
                    max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (1024 ** 3)
                    torch.cuda.reset_peak_memory_stats(device=self.device)
                    if self.is_main_process:
                        print(f"[Batch {batch_idx + 1}] Peak GPU Memory: {max_mem_gb:.2f} GB", flush=True)
                elif 'xpu' in str(self.device):
                    max_mem_gb = torch.xpu.max_memory_allocated(device=self.device) / (1024 ** 3)
                    if self.is_main_process:
                        print(f"[Batch {batch_idx + 1}] Peak GPU Memory: {max_mem_gb:.2f} GB", flush=True)
                    torch.xpu.reset_peak_memory_stats(device=self.device)

            train_time = train_end_time - move_data
            data_move_time = fwd_pass_st - move_data
            fwd_pass_time = fwd_pass_end - fwd_pass_st
            loss_time = loss_end - loss_st
            bwd_pass_time = bwd_pass_end - bwd_pass_st
            opt_step_time = opt_step_end - opt_step_st
            other_time = train_time - (data_move_time + fwd_pass_time + loss_time + bwd_pass_time + opt_step_time)
            if timing_window_start <= batch_idx <= timing_window_end:
                io_time_sum += io_time
                data_move_time_sum += data_move_time
                fwd_pass_time_sum += fwd_pass_time
                loss_time_sum += loss_time
                bwd_pass_time_sum += bwd_pass_time
                opt_step_time_sum += opt_step_time
                train_time_sum += train_time
                other_time_sum += other_time
                timing_samples += 1
                if batch_idx == timing_window_end:
                    avg_io_time = io_time_sum / timing_samples
                    avg_data_move_time = data_move_time_sum / timing_samples
                    avg_fwd_pass_time = fwd_pass_time_sum / timing_samples
                    avg_loss_time = loss_time_sum / timing_samples
                    avg_bwd_pass_time = bwd_pass_time_sum / timing_samples
                    avg_opt_step_time = opt_step_time_sum / timing_samples
                    avg_train_time = train_time_sum / timing_samples
                    avg_other_time = other_time_sum / timing_samples
                    if self.is_main_process:
                        print(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                            f"[Avg Timing Batches {timing_window_start + 1}-{timing_window_end + 1}] "
                            f"loader_wait: {avg_io_time:.3f}s | data_move: {avg_data_move_time:.3f}s | "
                            f"fwd: {avg_fwd_pass_time:.3f}s | loss: {avg_loss_time:.3f}s | "
                            f"bwd: {avg_bwd_pass_time:.3f}s | opt: {avg_opt_step_time:.3f}s | "
                            f"other: {avg_other_time:.3f}s | total_train: {avg_train_time:.3f}s",
                            flush=True,
                        )
                    #break
                    #import sys
                    #sys.exit(0)
            
            running_loss += loss.detach().item()

            # Also track the amplitude and phase loss to see if the network is predicting something reasonable
            # Note: With CUDAPrefetcher, amp_patch and ph_patch are already on GPU, so no need to move to CPU
            loss_amp = criterion(output_amp.detach(), amp_patch)
            loss_ph = criterion(output_ph.detach(), ph_patch)
            running_amp_loss += loss_amp.item()
            running_ph_loss += loss_ph.item()

            # Print progress more frequently for debugging (every 100 batches for first 1000, then every 1000)
            if batch_idx > 0 and self.is_main_process:
                from datetime import datetime
                if batch_idx < 1000 and (batch_idx + 1) % 100 == 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Training] Batch {batch_idx + 1}/{total_batches} ({(batch_idx + 1) * 100.0 / total_batches:.2f}%)", flush=True)
                    print(f"Training Running loss: amp: {running_amp_loss}, ph: {running_ph_loss}", flush=True)
                elif (batch_idx + 1) % 1000 == 0:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Training] Batch {batch_idx + 1}/{total_batches} ({(batch_idx + 1) * 100.0 / total_batches:.2f}%)", flush=True)
            
            if next_milestone_idx < len(milestone_batches) and batch_idx >= milestone_batches[next_milestone_idx]:
                from datetime import datetime
                progress_pct = int(progress_milestones[next_milestone_idx] * 100)
                if self.is_main_process:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Training Progress] {progress_pct}% complete ({batch_idx + 1}/{total_batches} batches)", flush=True)
                next_milestone_idx += 1

        if profiling:
            prof.stop()
        # Calculate average losses (use len(dataloader) for batch count)
        num_batches = len(dataloader)
        avg_train_loss = running_loss / num_batches
        avg_amp_loss = running_amp_loss / num_batches
        avg_ph_loss = running_ph_loss / num_batches
        
        if self.is_main_process:
            from datetime import datetime
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training epoch complete: {num_batches} batches processed", flush=True)

        #print(f"Training loop: before sync: {avg_train_loss}")
        # Synchronize losses across all ranks for DDP
        avg_train_loss = self.synchronize_loss(avg_train_loss)
        avg_amp_loss = self.synchronize_loss(avg_amp_loss)
        avg_ph_loss = self.synchronize_loss(avg_ph_loss)
        #print(f"Training loop: after sync: {avg_train_loss}")

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
        if self.is_main_process:
            from datetime import datetime
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting validation loop: {len(dataloader)} batches", flush=True)
        
        val_loss = 0.0
        val_amp_loss = 0.0
        val_ph_loss = 0.0
        
        # Progress milestones (every 10%)
        total_batches = len(dataloader)
        progress_milestones = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        milestone_batches = [int(m * total_batches) for m in progress_milestones]
        next_milestone_idx = 0

        # Variables for plotting (save last batch)
        last_input_diff = None
        last_output_diff = None
        last_amp_patch = None
        last_output_amp = None
        last_ph_patch = None
        last_output_ph = None

        # Use no_grad() to prevent gradient computation during validation
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Unpack batch
                diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch

                input_diff = diff_amp.to(self.device)
                input_probe = torch.view_as_real(probe.clone().detach()).to(self.device)
                input_norm = norm.to(self.device)
                input_scale = scale.to(self.device)
                if self.use_prefetch == False:
                    ph_patch = ph_patch.to(self.device)
                    amp_patch = amp_patch.to(self.device)
                output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)

                if self.mode == 'supervised':
                    loss = criterion(output_amp, amp_patch.to(self.device)) + criterion(output_ph, ph_patch.to(self.device))
                else:
                    loss = criterion(output_diff, input_diff)
                    #loss = criterion(torch.log10(output_diff + 1.0e-6), torch.log10(input_diff + 1.0e-6))
                val_loss += loss.detach().item()

                # Note: With CUDAPrefetcher, amp_patch and ph_patch are already on GPU, so no need to move to CPU
                loss_amp = criterion(output_amp.detach(), amp_patch)
                loss_ph = criterion(output_ph.detach(), ph_patch)
                val_amp_loss += loss_amp.item()
                val_ph_loss += loss_ph.item()

                # Print progress milestones
                if next_milestone_idx < len(milestone_batches) and batch_idx >= milestone_batches[next_milestone_idx]:
                    from datetime import datetime
                    progress_pct = int(progress_milestones[next_milestone_idx] * 100)
                    if self.is_main_process:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Validation Progress] {progress_pct}% complete ({batch_idx + 1}/{total_batches} batches)", flush=True)
                    next_milestone_idx += 1

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
        
        if self.is_main_process:
            from datetime import datetime
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation epoch complete: {num_batches} batches processed avg_val_loss:{avg_val_loss}", flush=True)

        # Synchronize losses across all ranks for DDP
        avg_val_loss = self.synchronize_loss(avg_val_loss)
        avg_val_amp_loss = self.synchronize_loss(avg_val_amp_loss)
        avg_val_ph_loss = self.synchronize_loss(avg_val_ph_loss)

        #print(f"sync'd loss avg_val_loss: {avg_val_loss} ")
        # Log and save metrics (synchronized values)
        if self.is_main_process and self.wandb_enabled:
            wandb.log({"val_loss": avg_val_loss})
            wandb.log({"val_amp_loss": avg_val_amp_loss})
            wandb.log({"val_ph_loss": avg_val_ph_loss})

        #print(f" after sync'd loss avg_val_loss: {avg_val_loss} ")
        metrics['validation_loss'].append(avg_val_loss)
        metrics['val_amp_loss'].append(avg_val_amp_loss)
        metrics['val_ph_loss'].append(avg_val_ph_loss)

        #print(f" 2 after sync'd loss avg_val_loss: {avg_val_loss} ")

        if plot and self.is_main_process and last_input_diff is not None:
            #print(f"in the plot func  ")
            # Extract first item from batch (handle batch_size=1 case properly)
            # Don't use squeeze() on the batch dimension to avoid removing it when batch_size=1
            input_diff = last_input_diff[0, 0].detach().cpu().numpy()  # [B, C, H, W] -> [H, W]
            output_diff = last_output_diff[0, 0].detach().cpu().numpy()  # [B, C, H, W] -> [H, W]
            # Note: With CUDAPrefetcher, tensors are on GPU, so move to CPU for plotting
            input_amp = last_amp_patch[0, 0].cpu()  # [B, C, H, W] -> [H, W]
            output_amp = last_output_amp[0, 0].detach().cpu().numpy()  # [B, C, H, W] -> [H, W]
            input_ph = last_ph_patch[0, 0].cpu()  # [B, C, H, W] -> [H, W]
            output_ph = last_output_ph[0, 0].detach().cpu().numpy()  # [B, C, H, W] -> [H, W]
            print(f"input_diff:{input_diff.shape}")
            filename = 'plot_epoch' + str(epoch) + '.png'
            print("generate plot for validation")
            self.generate_plot(input_diff, output_diff, input_amp, output_amp, input_ph, output_ph, filename)
            if self.wandb_enabled:
                run_path = os.path.join(self.model_save_path, 'run' + str(self.run_num))
                wandb.log({"val_plot": wandb.Image(os.path.join(run_path, filename), caption=f"Epoch {epoch}")})

        if scheduler:
            scheduler.step(avg_val_loss)
            metrics['lr'].append(optimizer.param_groups[0]['lr'])
            if self.is_main_process and self.wandb_enabled:
                wandb.log({"lr": optimizer.param_groups[0]['lr']})
        print(f"eval loss")
        # Check if this is the best model (use synchronized validation loss)
        # Only save on main process to avoid multiple saves
        if avg_val_loss < metrics['best_val_loss']:
            if self.is_main_process:
                print("Saving improved model after Val. Loss improved from %.4f to %.5f"
                      % (metrics['best_val_loss'], avg_val_loss), flush=True)
                self.update_saved_model('best_model')
            # Update best_val_loss on all ranks so they stay in sync
            metrics['best_val_loss'] = avg_val_loss
