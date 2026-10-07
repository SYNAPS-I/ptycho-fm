import os
import random
import shutil
import time
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from matplotlib import colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from ptychi.image_proc import place_patches_fourier_shift
from torch import nn

from ptycho_fm.metrics import (
    compute_psnr,
    compute_ssim,
    prepare_image_metric_inputs,
)
from ptycho_fm.utils.progress import TrainingProgress


def _move_to_cpu(obj):
    # Recursively detach and move Torch tensors to CPU (for safe serialization).
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [_move_to_cpu(v) for v in obj]
        return tuple(moved) if isinstance(obj, tuple) else moved
    return obj


def _atomic_torch_save(obj, path):
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _checkpoint_resume_context(context):
    """Remove execution-only settings that do not change training semantics."""
    normalized = dict(context)
    data = context.get('data')
    if isinstance(data, dict):
        normalized['data'] = {
            key: value for key, value in data.items() if key != 'num_workers'
        }
    return normalized


class Trainer:
    def __init__(
        self,
        model,
        mode,
        run_name,
        device,
        model_save_path,
        is_main_process=True,
        use_ddp=False,
        debug_mode=False,
        skip_batch_if_grad_norm_greater_than=None,
        experiment_logger=None,
        progress=None,
        resume_context=None,
    ):
        super().__init__()
        self.model = model
        self.progress = progress or TrainingProgress()
        self.resume_context = resume_context or {}
        self.mode = mode
        self.run_name = run_name
        self.device = device
        self.model_save_path = model_save_path
        self.is_main_process = is_main_process
        self.use_ddp = use_ddp
        self.debug_mode = debug_mode
        self.skip_batch_if_grad_norm_greater_than = skip_batch_if_grad_norm_greater_than
        self.experiment_logger = experiment_logger

    def _range_push(self, name):
        if self.device.type == 'cuda':
            torch.cuda.nvtx.range_push(name)

    def _range_pop(self):
        if self.device.type == 'cuda':
            torch.cuda.nvtx.range_pop()

    def synchronize_loss(self, loss_value):
        """Synchronize loss across all processes in DDP."""
        if self.use_ddp and dist.is_initialized():
            loss_tensor = torch.tensor(loss_value, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor /= dist.get_world_size()
            return loss_tensor.item()
        return loss_value

    @staticmethod
    def compute_loss(
        criterion, input, target, probe=None, normalization=None, scale=None
    ):
        """Call losses that optionally require the probe far-field envelope."""
        if getattr(criterion, 'uses_probe_envelope', False):
            return criterion(
                input,
                target,
                probe=probe,
                normalization=normalization,
                scale=scale,
            )
        return criterion(input, target)

    def update_saved_model(self, name):
        """Update saved model (checkpoints and if validation loss is minimized)."""
        os.makedirs(self.model_save_path, exist_ok=True)
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_name))
        os.makedirs(run_path, exist_ok=True)

        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        # Save a CPU copy so serialization does not depend on CUDA contexts/devices.
        cpu_state_dict = _move_to_cpu(state_dict)
        _atomic_torch_save(cpu_state_dict, os.path.join(run_path, name + '.pth'))

    def save_config(self, config_path='config.yaml'):
        """Save a copy of the config file to the run path for reproducibility."""
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_name))
        if not os.path.isdir(run_path):
            os.makedirs(run_path, exist_ok=True)
        if os.path.exists(config_path):
            shutil.copy(config_path, os.path.join(run_path, 'config.yaml'))

    def _rank_runtime(self):
        numpy_state = np.random.get_state()
        return {
            'progress': self.progress.state_dict(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(self.device) if self.device.type == 'cuda' else None,
            'python_rng': random.getstate(),
            'numpy_rng': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        }

    def _optimizer_signature(self, optimizer):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        names = {id(param): name for name, param in model.named_parameters()}
        return [[(names[id(param)], tuple(param.shape)) for param in group['params']]
                for group in optimizer.param_groups]

    def generate_state_dict(self, epoch_num, metrics, optimizer, scheduler=None):
        """All ranks participate so resume restores each rank's RNG and interval."""
        local = self._rank_runtime()
        runtime = [local]
        if self.use_ddp and dist.is_initialized():
            runtime = [None] * dist.get_world_size()
            dist.all_gather_object(runtime, local)
        return {
            'format_version': 2,
            'current_epoch': self.progress.epoch,
            'loss_tracker': metrics,
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_signature': self._optimizer_signature(optimizer),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'rank_runtime': runtime,
            'resume_context': self.resume_context,
        }

    def save_model_and_states_checkpoint(
        self, epoch_num, metrics, optimizer=None, scheduler=None
    ):
        state = self.generate_state_dict(epoch_num, metrics, optimizer, scheduler)
        if not self.is_main_process:
            return
        state_path = os.path.join(self.model_save_path, 'run' + str(self.run_name))
        os.makedirs(state_path, exist_ok=True)
        self.update_saved_model('checkpoint_model')
        if self.progress.enabled:
            self.update_saved_model(f'model_iters_{self.progress.iters}')
            _atomic_torch_save(_move_to_cpu(state), os.path.join(state_path, f'state_iters_{self.progress.iters}.pth'))
        else:
            self.update_saved_model(f'model_epoch_{epoch_num + 1:03d}')
        _atomic_torch_save(_move_to_cpu(state), os.path.join(state_path, 'checkpoint.state'))

    def load_state_checkpoint(self, optimizer, scheduler=None, checkpoint_file=None,
                              skip_scheduler=False):
        checkpoint_fname = checkpoint_file or os.path.join(
            self.model_save_path, 'run' + str(self.run_name), 'checkpoint.state')
        state = torch.load(checkpoint_fname, map_location='cpu', weights_only=True)
        signature = state.get('optimizer_signature')
        if signature is not None and signature != self._optimizer_signature(optimizer):
            raise ValueError('Optimizer parameters changed; explicitly convert the checkpoint or use weights-only fine-tuning')
        if state.get('format_version') != 2:
            if self.progress.enabled or 'iters' in state:
                raise ValueError('Legacy iteration/compute state requires explicit conversion before resume; weights-only loading is fine-tuning')
            self.progress.epoch = state['current_epoch']
        else:
            saved_context = _checkpoint_resume_context(state['resume_context'])
            active_context = _checkpoint_resume_context(self.resume_context)
            if saved_context != active_context:
                raise ValueError('Resume dataset/model/batch/world-size settings differ from the checkpoint')
            rank = dist.get_rank() if self.use_ddp and dist.is_initialized() else 0
            runtime = state['rank_runtime'][rank]
            self.progress.load_state_dict(runtime['progress'])
            torch.set_rng_state(runtime['torch_rng'])
            if runtime['cuda_rng'] is not None and self.device.type == 'cuda':
                torch.cuda.set_rng_state(runtime['cuda_rng'], self.device)
            random.setstate(runtime['python_rng'])
            numpy_state = runtime['numpy_rng']
            np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
        try:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        except ValueError as exc:
            raise ValueError('Optimizer groups require explicit conversion; use weights-only fine-tuning for a fresh optimizer') from exc
        for param, param_state in optimizer.state.items():
            for key, value in param_state.items():
                if torch.is_tensor(value):
                    if value.numel() > 1 and value.shape != param.shape:
                        raise ValueError('Optimizer tensor shapes require explicit checkpoint conversion')
                    param_state[key] = value.to(self.device)
        if not skip_scheduler and state.get('scheduler_state_dict') is not None:
            if scheduler is None:
                raise ValueError('Checkpoint contains a scheduler; resume requires the same scheduler configuration')
            scheduler.load_state_dict(state['scheduler_state_dict'])
        return self.progress.epoch, state['loss_tracker'], optimizer, scheduler

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
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_name))
        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        f.savefig(os.path.join(run_path, filename), bbox_inches='tight', transparent=True)

    def generate_test_plot(
        self, dataloader, epoch, filename, central_crop=64, object_crop=180,
        log_key="test_plot", artifact_path="test_plots", caption=None,
    ):
        total_scan_points = len(dataloader.dataset)
        pred_amp = torch.zeros((total_scan_points, dataloader.dataset.pattern_shape[0], dataloader.dataset.pattern_shape[1]), device='cpu')
        pred_ph = torch.zeros(pred_amp.shape, device='cpu')
        gt_amp = torch.zeros(pred_amp.shape, device='cpu')
        gt_ph = torch.zeros(pred_amp.shape, device='cpu')
        scan_idx = 0

        with torch.no_grad():
            inference_model = self.model.module if self.use_ddp else self.model
            for i, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)) and len(batch) == 8:
                    diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale, _meta = batch
                else:
                    diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale = batch
                batch_size = diff_amp.size(0)

                input_diff = diff_amp.to(self.device, non_blocking=True)
                input_probe = torch.view_as_real(probe.clone().detach()).to(self.device, non_blocking=True)
                input_norm = norm.to(self.device, non_blocking=True)
                input_scale = scale.to(self.device, non_blocking=True)

                _output_diff, output_amp, output_ph = inference_model(input_diff, input_probe, input_norm, input_scale)
                pred_amp[scan_idx:scan_idx + batch_size] = output_amp.squeeze().detach().cpu()
                pred_ph[scan_idx:scan_idx + batch_size] = output_ph.squeeze().detach().cpu()
                gt_amp[scan_idx:scan_idx + batch_size] = amp_patch.squeeze().detach().cpu()
                gt_ph[scan_idx:scan_idx + batch_size] = ph_patch.squeeze().detach().cpu()

                scan_idx += batch_size

        object_size = dataloader.dataset.object_shape
        positions = dataloader.dataset.get_probe_positions()
        pred_amp_object = torch.zeros(object_size, device='cpu')
        pred_ph_object = torch.zeros(object_size, device='cpu')
        buffer = torch.zeros(object_size, device='cpu')

        central_crop = int(central_crop)
        if central_crop < 0 or 2 * central_crop >= min(dataloader.dataset.pattern_shape):
            raise ValueError("central_crop must be non-negative and smaller than half the pattern size.")
        patch_slice = slice(central_crop, -central_crop if central_crop else None)
        pred_amp_object = place_patches_fourier_shift(
            pred_amp_object,
            positions,
            pred_amp[:, patch_slice, patch_slice],
            op="add",
            adjoint_mode=False,
            pad=32
        )
        pred_ph_object = place_patches_fourier_shift(
            pred_ph_object,
            positions,
            pred_ph[:, patch_slice, patch_slice],
            op="add",
            adjoint_mode=False,
            pad=32
        )
        buffer = place_patches_fourier_shift(
            buffer,
            positions,
            torch.ones_like(pred_ph[:, patch_slice, patch_slice]),
            op="add",
            adjoint_mode=False,
            pad=32
        )

        gt_amp_object = torch.zeros(object_size, device='cpu')
        gt_ph_object = torch.zeros(object_size, device='cpu')
        gt_amp_object = place_patches_fourier_shift(
            gt_amp_object,
            positions,
            gt_amp[:, patch_slice, patch_slice],
            op="add",
            adjoint_mode=False,
            pad=32
        )
        gt_ph_object = place_patches_fourier_shift(
            gt_ph_object,
            positions,
            gt_ph[:, patch_slice, patch_slice],
            op="add",
            adjoint_mode=False,
            pad=32
        )

        pred_amp_object = pred_amp_object / torch.clip(buffer, min=1)
        pred_ph_object = pred_ph_object / torch.clip(buffer, min=1)
        gt_amp_object = gt_amp_object / torch.clip(buffer, min=1)
        gt_ph_object = gt_ph_object / torch.clip(buffer, min=1)

        object_crop = int(object_crop)
        if object_crop < 0 or 2 * object_crop >= min(object_size):
            raise ValueError("object_crop must be non-negative and smaller than half the object size.")
        object_slice = slice(object_crop, -object_crop if object_crop else None)
        phase_region = pred_ph_object[object_slice, object_slice]
        vmin_ph = torch.mean(phase_region) - (2 * torch.std(phase_region))
        vmax_ph = torch.mean(phase_region) + (2 * torch.std(phase_region))

        f, ax = plt.subplots(figsize=(9, 8), ncols=2, nrows=2)

        gt0 = ax[0, 0].imshow(gt_amp_object[object_slice, object_slice], interpolation='none', vmin=0.9, vmax=1)
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(gt0, cax=cax, orientation='vertical')
        ax[0, 0].set_title('LSQML amplitude')

        gt1 = ax[0, 1].imshow(gt_ph_object[object_slice, object_slice], interpolation='none', vmin=-1.3, vmax=1.3, cmap='magma')
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(gt1, cax=cax, orientation='vertical')
        ax[0, 1].set_title('LSQML phase')

        pred0 = ax[1, 0].imshow(pred_amp_object[object_slice, object_slice], interpolation='none', vmin=0.9, vmax=1)
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(pred0, cax=cax, orientation='vertical')
        ax[1, 0].set_title('Predicted amplitude')

        pred1 = ax[1, 1].imshow(pred_ph_object[object_slice, object_slice], interpolation='none', vmin=vmin_ph, vmax=vmax_ph, cmap='magma')
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(pred1, cax=cax, orientation='vertical')
        ax[1, 1].set_title('Predicted phase')

        plt.tight_layout()
        run_path = os.path.join(self.model_save_path, 'run' + str(self.run_name))
        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        f.savefig(os.path.join(run_path, filename), bbox_inches='tight', transparent=True)
        plt.close(f)

        if self.experiment_logger is not None:
            self.experiment_logger.log_image(
                log_key,
                os.path.join(run_path, filename),
                caption=caption or f"Test: epoch {epoch}",
                step=self.progress.iters,
                artifact_path=artifact_path,
            )

    def train(self, dataloader, criterion, optimizer, metrics, profile=False, epoch=None,
              scheduler=None, scheduler_unit="epoch", on_iteration=None):
        """
        Training loop.

        Args:
            dataloader: PyTorch DataLoader yielding batches of
                       (diff_amp, amp_patch, ph_patch, probe, probe_pos, norm, scale[, meta])
        """
        self.progress.epoch = epoch or 0
        if self.progress.stopped:
            return
        running_loss = 0.0
        running_amp_loss = 0.0
        running_ph_loss = 0.0
        processed_batches = 0

        total_batches = len(dataloader)
        progress_milestones = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        milestone_batches = [int(total_batches * m) for m in progress_milestones]
        next_milestone_idx = 0


        if self.is_main_process:
            print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] Starting training loop: {total_batches} total batches", flush=True)
            print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] Loading first batch... (this may take a while with lazy data loading)", flush=True)

        batch_0_io_start = time.time()
        train_end_time = None  # Will be set after each batch completes
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # adding some profiling
            if profile:
                if epoch == 1 and batch_idx == 0 and self.is_main_process:
                    torch.cuda.profiler.start()
                if batch_idx == 50:
                    if epoch == 1 and self.is_main_process:
                        torch.cuda.profiler.stop()
                    break

            self._range_push(f"step {batch_idx}")
            self._range_push(f"data copy in {batch_idx}")

            # Time IO (data loading) - this captures the time to get batch from DataLoader
            # The DataLoader fetch happens at the 'for' line above, so we time from end of previous batch
            if batch_idx < 10:
                if batch_idx == 0:
                    io_time = time.time() - batch_0_io_start
                else:
                    io_time = time.time() - train_end_time

            if batch_idx == 0 and self.is_main_process:
                print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] First batch loaded! Starting training... (batch 1/{total_batches})", flush=True)
            elif batch_idx < 10 and self.is_main_process:
                print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] Processing batch {batch_idx + 1}/{total_batches}", flush=True)

            # ── Unpack ──────────────────────────────────────────────────────────
            if isinstance(batch, (list, tuple)) and len(batch) == 8:
                diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale, _meta = batch
            else:
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

            self._range_pop()  # copy in

            self._range_push("forward")

            output_diff, output_amp, output_ph = self.model(input_diff, input_probe, input_norm, input_scale)

            if self.mode == 'supervised':
                # amp_patch/ph_patch already on device
                loss = self.compute_loss(
                    criterion, output_amp, amp_patch
                ) + self.compute_loss(criterion, output_ph, ph_patch)
            else:
                loss = self.compute_loss(
                    criterion,
                    output_diff,
                    input_diff,
                    input_probe,
                    input_norm,
                    input_scale,
                )

            self._range_pop()  # forward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            total_norm = None
            global_max_norm = None
            need_grad_norm = (
                self.debug_mode
                or (self.skip_batch_if_grad_norm_greater_than is not None)
            )
            if need_grad_norm:
                total_norm_sq = 0.0
                for param in self.model.parameters():
                    if param.grad is None:
                        continue
                    grad_norm = param.grad.detach().float().norm(2)
                    total_norm_sq += float(grad_norm) ** 2
                total_norm = total_norm_sq ** 0.5
                global_max_norm = total_norm

                if self.use_ddp and dist.is_initialized() and self.skip_batch_if_grad_norm_greater_than is not None:
                    norm_tensor = torch.tensor(total_norm, device=self.device)
                    dist.all_reduce(norm_tensor, op=dist.ReduceOp.MAX)
                    global_max_norm = float(norm_tensor.item())

            updated = False
            skip_threshold = self.skip_batch_if_grad_norm_greater_than
            if skip_threshold is not None and global_max_norm is not None and global_max_norm > skip_threshold:
                if self.is_main_process:
                    print(
                        f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"[Skip Batch] Grad norm {global_max_norm:.2f} > {skip_threshold} "
                        f"(batch {batch_idx + 1}/{total_batches})",
                        flush=True
                    )
            else:
                self._range_push("optimizer")
                optimizer.step()
                updated = True
                if scheduler is not None and scheduler_unit == "update":
                    scheduler.step()
                self._range_pop()  # optimizer
                processed_batches += 1

            train_end_time = time.time()

            if batch_idx < 10:
                train_time = train_end_time - train_start_time
                total_time = io_time + train_time
                if self.is_main_process:
                    print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] [Batch {batch_idx + 1} Timing] IO: {io_time:.3f}s | Training: {train_time:.3f}s | Total: {total_time:.3f}s", flush=True)

            batch_loss = loss.detach().item()
            running_loss += batch_loss

            # Track amp/phase losses (targets now guaranteed on same device)
            loss_amp = self.compute_loss(criterion, output_amp.detach(), amp_patch)
            loss_ph = self.compute_loss(criterion, output_ph.detach(), ph_patch)
            batch_amp_loss = loss_amp.item()
            batch_ph_loss = loss_ph.item()
            running_amp_loss += batch_amp_loss
            running_ph_loss  += batch_ph_loss


            if (
                batch_idx > 0
                and self.is_main_process
                and ((batch_idx < 1000 and (batch_idx + 1) % 1000 == 0) or (batch_idx + 1) % 1000 == 0)
            ):
                print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] [Training] Batch {batch_idx + 1}/{total_batches} ({(batch_idx + 1) * 100.0 / total_batches:.2f}%)", flush=True)

            if next_milestone_idx < len(milestone_batches) and batch_idx >= milestone_batches[next_milestone_idx]:
                progress_pct = int(progress_milestones[next_milestone_idx] * 100)
                if self.is_main_process:
                    print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] [Training Progress] {progress_pct}% complete ({batch_idx + 1}/{total_batches} batches)", flush=True)
                next_milestone_idx += 1


            self._range_pop()  # step

            batch_size = int(diff_amp.size(0))
            tflops = (self.progress.calculator.training_tflops(batch_size)
                      if self.progress.calculator else 0.0)
            totals = torch.tensor([batch_size, tflops], dtype=torch.float64, device=self.device)
            if self.use_ddp and dist.is_initialized():
                dist.all_reduce(totals)
            due, crossed = self.progress.record(
                batch_size=batch_size, global_samples=int(totals[0].item()),
                tflops=float(totals[1].item()), updated=updated,
                losses=(batch_loss, batch_amp_loss, batch_ph_loss),
            )

            # Log only after record() advances the completed-iteration count.
            if (
                self.debug_mode
                and self.experiment_logger is not None
                and self.is_main_process
            ):
                self.experiment_logger.log_metrics(
                    {'grad_norm': global_max_norm},
                    step=self.progress.iters,
                )

            batch_log_interval = (
                self.experiment_logger.log_every_n_batches
                if self.experiment_logger is not None
                else None
            )
            if (
                batch_log_interval is not None
                and self.is_main_process
                and self.progress.iters % batch_log_interval == 0
            ):
                batch_metrics = {
                    "train_batch_loss": batch_loss,
                    "train_batch_amp_loss": batch_amp_loss,
                    "train_batch_ph_loss": batch_ph_loss,
                }
                if self.device.type == 'cuda':
                    batch_metrics["train_gpu_mem_gb"] = (
                        torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                    )
                self.experiment_logger.log_metrics(
                    batch_metrics, step=self.progress.iters
                )

            at_boundary = batch_idx + 1 == total_batches
            if at_boundary:
                self.progress.finish_epoch()
            if on_iteration is not None and (due or at_boundary):
                on_iteration(epoch, crossed, at_boundary)
            if self.progress.stopped:
                break

        if self.progress.enabled:
            return
        if self.device.type == 'cuda':
            torch.cuda.synchronize()  # device sync to ensure accurate epoch timings
        print(f"Processed {processed_batches} batches", flush=True)
        num_batches = processed_batches
        if num_batches == 0:
            if self.is_main_process:
                print("[Warning] No training batches were processed.", flush=True)
            avg_train_loss = float("inf")
            avg_amp_loss = float("inf")
            avg_ph_loss = float("inf")
        else:
            avg_train_loss = running_loss / num_batches
            avg_amp_loss = running_amp_loss / num_batches
            avg_ph_loss = running_ph_loss / num_batches

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        if self.is_main_process:
            print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] Training epoch complete: {num_batches} batches processed in {epoch_time:.3f} seconds", flush=True)

        avg_train_loss = self.synchronize_loss(avg_train_loss)
        avg_amp_loss = self.synchronize_loss(avg_amp_loss)
        avg_ph_loss = self.synchronize_loss(avg_ph_loss)

        if self.experiment_logger is not None and self.is_main_process:
            self.experiment_logger.log_metrics(
                {
                    "train_epoch_loss": avg_train_loss,
                    "train_epoch_amp_loss": avg_amp_loss,
                    "train_epoch_ph_loss": avg_ph_loss,
                    "train_epoch_time_s": epoch_time,
                    "epoch": epoch,
                },
                step=self.progress.iters,
            )

        metrics['training_loss'].append(avg_train_loss)
        metrics['train_amp_loss'].append(avg_amp_loss)
        metrics['train_ph_loss'].append(avg_ph_loss)

    def validate(self, dataloader, criterion, optimizer, metrics, plot=False, epoch=0, scheduler=None, profile=False, emit_scalars=True):
        """
        Validation loop with SSIM and PSNR metrics.
        """
        if self.is_main_process:
            print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] Starting validation loop: {len(dataloader)} batches", flush=True)

        val_loss = 0.0
        val_amp_loss = 0.0
        val_ph_loss = 0.0

        # SSIM and PSNR accumulators
        val_amp_ssim = 0.0
        val_amp_psnr = 0.0
        val_ph_ssim = 0.0
        val_ph_psnr = 0.0
        num_samples = 0
        processed_batches = 0

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
                if profile and batch_idx == 50:
                    # early stop if profiling
                    break
                if isinstance(batch, (list, tuple)) and len(batch) == 8:
                    diff_amp, amp_patch, ph_patch, probe, _probe_pos, norm, scale, _meta = batch
                else:
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
                    loss = self.compute_loss(
                        criterion, output_amp, amp_patch
                    ) + self.compute_loss(criterion, output_ph, ph_patch)
                else:
                    loss = self.compute_loss(
                        criterion,
                        output_diff,
                        input_diff,
                        input_probe,
                        input_norm,
                        input_scale,
                    )

                val_loss += loss.detach().item()

                loss_amp = self.compute_loss(
                    criterion, output_amp.detach(), amp_patch
                )
                loss_ph = self.compute_loss(
                    criterion, output_ph.detach(), ph_patch
                )
                val_amp_loss += loss_amp.item()
                val_ph_loss += loss_ph.item()
                processed_batches += 1

                # Align each prediction to its target's global mean before scoring.
                batch_size = output_amp.size(0)

                pred_amp = output_amp.detach()
                pred_ph = output_ph.detach()

                for sample_pred_amp, sample_amp, sample_pred_ph, sample_ph in zip(
                    pred_amp, amp_patch, pred_ph, ph_patch, strict=True
                ):
                    aligned_amp, target_amp, amp_range = prepare_image_metric_inputs(
                        sample_pred_amp, sample_amp, align_mean=True
                    )
                    aligned_ph, target_ph, ph_range = prepare_image_metric_inputs(
                        sample_pred_ph, sample_ph, align_mean=True
                    )

                    val_amp_ssim += compute_ssim(
                        aligned_amp, target_amp, data_range=amp_range
                    )
                    val_ph_ssim += compute_ssim(
                        aligned_ph, target_ph, data_range=ph_range
                    )
                    val_amp_psnr += compute_psnr(
                        aligned_amp, target_amp, data_range=amp_range
                    )
                    val_ph_psnr += compute_psnr(
                        aligned_ph, target_ph, data_range=ph_range
                    )

                num_samples += batch_size

                if next_milestone_idx < len(milestone_batches) and batch_idx >= milestone_batches[next_milestone_idx]:
                    progress_pct = int(progress_milestones[next_milestone_idx] * 100)
                    if self.is_main_process:
                        print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] [Validation Progress] {progress_pct}% complete ({batch_idx + 1}/{total_batches} batches)", flush=True)
                    next_milestone_idx += 1

                if plot and self.is_main_process:
                    last_input_diff = input_diff
                    last_output_diff = output_diff
                    last_amp_patch = amp_patch
                    last_output_amp = output_amp
                    last_ph_patch = ph_patch
                    last_output_ph = output_ph

        num_batches = processed_batches
        if num_batches == 0:
            if self.is_main_process:
                print("[Warning] No validation batches were processed.", flush=True)
            avg_val_loss = float("inf")
            avg_val_amp_loss = float("inf")
            avg_val_ph_loss = float("inf")
        else:
            avg_val_loss = val_loss / num_batches
            avg_val_amp_loss = val_amp_loss / num_batches
            avg_val_ph_loss = val_ph_loss / num_batches
        
        # Average SSIM and PSNR
        avg_amp_ssim = val_amp_ssim / num_samples if num_samples > 0 else 0.0
        avg_amp_psnr = val_amp_psnr / num_samples if num_samples > 0 else 0.0
        avg_ph_ssim = val_ph_ssim / num_samples if num_samples > 0 else 0.0
        avg_ph_psnr = val_ph_psnr / num_samples if num_samples > 0 else 0.0

        if self.is_main_process:
            print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] Validation epoch complete: {num_batches} batches processed", flush=True)
            print(f"[{datetime.now(tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] Metrics - Amp SSIM: {avg_amp_ssim:.4f}, Amp PSNR: {avg_amp_psnr:.2f} dB, Phase SSIM: {avg_ph_ssim:.4f}, Phase PSNR: {avg_ph_psnr:.2f} dB", flush=True)

        avg_val_loss = self.synchronize_loss(avg_val_loss)
        avg_val_amp_loss = self.synchronize_loss(avg_val_amp_loss)
        avg_val_ph_loss = self.synchronize_loss(avg_val_ph_loss)
        avg_amp_ssim = self.synchronize_loss(avg_amp_ssim)
        avg_amp_psnr = self.synchronize_loss(avg_amp_psnr)
        avg_ph_ssim = self.synchronize_loss(avg_ph_ssim)
        avg_ph_psnr = self.synchronize_loss(avg_ph_psnr)

        if (
            emit_scalars
            and self.experiment_logger is not None
            and self.is_main_process
        ):
            self.experiment_logger.log_metrics(
                {
                    "val_loss": avg_val_loss,
                    "val_amp_loss": avg_val_amp_loss,
                    "val_ph_loss": avg_val_ph_loss,
                    "val_amp_ssim": avg_amp_ssim,
                    "val_amp_psnr": avg_amp_psnr,
                    "val_ph_ssim": avg_ph_ssim,
                    "val_ph_psnr": avg_ph_psnr,
                    "epoch": epoch,
                },
                step=self.progress.iters,
            )

        metrics['validation_loss'].append(avg_val_loss)
        metrics['val_amp_loss'].append(avg_val_amp_loss)
        metrics['val_ph_loss'].append(avg_val_ph_loss)

        # Add SSIM/PSNR to metrics (initialize lists if not present)
        if 'val_amp_ssim' not in metrics:
            metrics['val_amp_ssim'] = []
        if 'val_amp_psnr' not in metrics:
            metrics['val_amp_psnr'] = []
        if 'val_ph_ssim' not in metrics:
            metrics['val_ph_ssim'] = []
        if 'val_ph_psnr' not in metrics:
            metrics['val_ph_psnr'] = []

        metrics['val_amp_ssim'].append(avg_amp_ssim)
        metrics['val_amp_psnr'].append(avg_amp_psnr)
        metrics['val_ph_ssim'].append(avg_ph_ssim)
        metrics['val_ph_psnr'].append(avg_ph_psnr)


        if plot and self.is_main_process and last_input_diff is not None:
            input_diff_np = last_input_diff[0, 0].detach().cpu().numpy()
            output_diff_np = last_output_diff[0, 0].detach().cpu().numpy()

            input_amp = last_amp_patch[0, 0].detach().cpu()
            output_amp = last_output_amp[0, 0].detach().cpu().numpy()

            input_ph = last_ph_patch[0, 0].detach().cpu()
            output_ph = last_output_ph[0, 0].detach().cpu().numpy()

            filename = 'plot_epoch' + str(epoch) + '.png'
            self.generate_plot(input_diff_np, output_diff_np, input_amp, output_amp, input_ph, output_ph, filename)
            run_path = os.path.join(self.model_save_path, 'run' + str(self.run_name))
            plot_path = os.path.join(run_path, filename)

            if self.experiment_logger is not None:
                self.experiment_logger.log_image(
                    'val_plot',
                    plot_path,
                    caption=f"Epoch {epoch}",
                    step=self.progress.iters,
                    artifact_path='val_plots',
                )

        if scheduler:
            # Only ReduceLROnPlateau expects a monitored metric; others use epoch progression.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
            if 'lr' not in metrics:
                metrics['lr'] = []
            metrics['lr'].append(optimizer.param_groups[0]['lr'])
            if (
                emit_scalars
                and self.is_main_process
                and self.experiment_logger is not None
            ):
                self.experiment_logger.log_metrics(
                    {"lr": optimizer.param_groups[0]['lr'], "epoch": epoch},
                    step=self.progress.iters,
                )

        if avg_val_loss < metrics['best_val_loss']:
            if self.is_main_process:
                print(
                    f"Saving improved model after Val. Loss improved from {metrics['best_val_loss']:.4f} to {avg_val_loss:.5f}",
                    flush=True,
                )
                self.update_saved_model('best_model')
            metrics['best_val_loss'] = avg_val_loss

        # Synchronize all processes after validation and potential model saving
        # This prevents CUDA/HDF5 conflicts when other ranks continue while rank 0 saves
        if self.use_ddp and dist.is_initialized():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            dist.barrier()

        return {
            'val_loss': avg_val_loss, 'val_amp_loss': avg_val_amp_loss,
            'val_ph_loss': avg_val_ph_loss, 'val_amp_ssim': avg_amp_ssim,
            'val_amp_psnr': avg_amp_psnr, 'val_ph_ssim': avg_ph_ssim,
            'val_ph_psnr': avg_ph_psnr,
        }
