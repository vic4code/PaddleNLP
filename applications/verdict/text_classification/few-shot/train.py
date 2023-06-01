# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import defaultdict
from dataclasses import dataclass, field

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from metric import MetricReport
from utils import load_local_dataset

from paddlenlp.prompt import (
    AutoTemplate,
    PromptModelForSequenceClassification,
    PromptTrainer,
    PromptTuningArguments,
    SoftVerbalizer,
)
from paddlenlp.trainer import EarlyStoppingCallback, PdArgumentParser
from paddlenlp.transformers import AutoModelForMaskedLM, AutoTokenizer
from paddlenlp.utils.log import logger

from paddlenlp.trainer.trainer import *
from paddlenlp.trainer.trainer_callback import *
from paddlenlp.transformers.model_utils import PretrainedModel, _add_variant, unwrap_model
from paddlenlp.prompt.prompt_utils import *

from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# yapf: disable
@dataclass
class DataArguments:
    data_dir: str = field(default="./data", metadata={"help": "The dataset dictionary includes train.txt, dev.txt and label.txt files."})
    prompt: str = field(default=None, metadata={"help": "The input prompt for tuning."})
    # dataloader_drop_last: bool = field(default=True, metadata={"help": "Drop the last for uncomplete batch"})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-3.0-base-zh", metadata={"help": "The build-in pretrained model or the path to local model."})
    export_type: str = field(default='paddle', metadata={"help": "The type to export. Support `paddle` and `onnx`."})
# yapf: enable


class PromptDataCollatorWithPadding(PromptDataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.default_model_input_names: List = (
            "input_ids",
            "token_type_ids",
            "special_tokens_mask",
            "offset_mapping",
            "position_ids",
            "id",
            "nth_chunk",
            "num_chunks"
        )

        batch = {}
        for key in features[0]:
            if key in self.default_model_input_names:
                batch[key] = [b[key] for b in features]

        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            return_attention_mask=self.return_attention_mask,
        )

        max_length = batch["input_ids"].shape[1]
        for key in features[0]:
            if key not in self.default_model_input_names:
                values = [b[key] for b in features if key in b]
                if len(values) < len(features):
                    continue
                if key == "masked_positions":
                    new_values = []
                    for index, value in enumerate(values):
                        value = np.array(value) + index * max_length
                        new_values.extend(value.tolist())
                    values = new_values
                elif key == "attention_mask":
                    new_values = np.ones([len(values), 1, max_length, max_length]) * -1e4
                    for index, value in enumerate(values):
                        length = len(value)
                        new_values[index][0, :length, :length] = value
                    values = new_values
                elif key in ("soft_token_ids", "encoder_ids"):
                    for index, value in enumerate(values):
                        values[index] = value + [0] * (max_length - len(value))
                elif key in ("omask_positions"):
                    max_num_option = max([len(x) for x in values])
                    for index, value in enumerate(values):
                        values[index] = value + [0] * (max_num_option - len(value))
                elif key == "labels":
                    if isinstance(values[0], list):
                        max_num_label = max([len(x) for x in values])
                        for index, value in enumerate(values):
                            values[index] = value + [-100] * (max_num_label - len(value))
                elif key != "cls_positions":
                    continue
                batch[key] = self._convert_to_tensors(values)
        
        return batch


class PromptTrainer(PromptTrainer):
    def _get_train_sampler(self) -> Optional[paddle.io.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.world_size <= 1:
            return paddle.io.BatchSampler(
                dataset=self.train_dataset,
                shuffle=False,
                batch_size=self.args.per_device_train_batch_size,
                drop_last=self.args.dataloader_drop_last,
            )

        return DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            num_replicas=self.args.dataset_world_size,
            rank=self.args.dataset_rank,
            drop_last=self.args.dataloader_drop_last,
        )
    

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~paddle.io.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`paddle.io.Dataset`, *optional*):
                The test dataset to use. If it is an `datasets.Dataset`, columns not accepted by the `model.forward()`
                method are automatically removed. It must implement `__len__`.
        """
        self.args.dataloader_drop_last = False
        test_dataset = self._map_dataset(test_dataset)
        
        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            test_dataset = self._remove_unused_columns(test_dataset, description="eval")

        if self._is_iterable_dataset(test_dataset):
            if self.args.dataset_world_size > 1:
                test_dataset = IterableDatasetShard(
                    test_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.dataset_world_size,
                    process_index=self.args.dataset_rank,
                )

            return DataLoader(
                test_dataset,
                batch_size=self.args.per_device_eval_batch_size * self.world_size,
                collate_fn=self.data_collator,  # _get_collator_with_removed_columns
                num_workers=self.args.dataloader_num_workers,
            )

        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )


    def compute_forward(self, model, inputs):
        """
        Compute the total loss for every batch.
        """

        # model outputs depend on the args
        logits = model(**inputs)
    
        return logits


    def compute_loss(self, labels, logits, return_outputs=True, is_train=False):
        """
        Compute the total loss for every batch in the same text id.
        """

        labels = labels[-1, :].unsqueeze(axis=0)
        loss = self.criterion(logits, labels)

        if is_train:
            logger.info(
                f"training_loss: {loss.item()}."
            )
        
        else:
            logger.info(
                f"eval_loss: {loss.item()}."
            )

        return loss


    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
    ):  
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
        """

        args = self.args
        self.is_in_train = True
        resume_from_checkpoint = None if not resume_from_checkpoint else resume_from_checkpoint

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if isinstance(self.model, LoRAModel):
                weight_name = LORA_WEIGHT_FILE_NAME
            elif isinstance(self.model, PrefixModelForCausalLM):
                weight_name = PREFIX_WEIGHT_FILE_NAME
            else:
                weight_name = PADDLE_WEIGHT_FILE_NAME
            if not os.path.isfile(
                os.path.join(resume_from_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix))
            ):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint} .")

            # TODO: Need to load the model state dict on the CPU to avoid an OOM error.
            state_dict = paddle.load(
                os.path.join(resume_from_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix)),
                return_numpy=True,
            )
            # If the model is on the GPU, it still works!
            self._set_state_dict_in_model(state_dict)

            # release memory
            del state_dict

        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.dataset_world_size
        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = len(self.train_dataset)

            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = int(len(self.train_dataset) * args.num_train_epochs)

            if args.minimum_eval_times is not None and args.minimum_eval_times > 0:
                if max_steps // args.eval_steps < args.minimum_eval_times:
                    exp_step = max_steps / args.minimum_eval_times
                    exp_step = max(int(exp_step - exp_step % 10), 10)
                    logger.info("Reset eval step by minimum_eval_times to %d" % exp_step)
                    args.eval_steps = exp_step
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                f"args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}"
            )

        # delay_optimizer_creation = (
        #     self.sharding is not None
        #     and ShardingOption.SHARD_OP in self.args.sharding
        # )
        delay_optimizer_creation = False

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(f"  Total num train samples = {num_train_samples}")
        # per_device_trainable_numel = sum(p.numel().item() for p in model.parameters() if not p.stop_gradient)
        # TODO: Temporary fix since Tensor.numel() not supported in distributed mode
        per_device_trainable_numel = sum(np.prod(p.shape) for p in model.parameters() if not p.stop_gradient)
        logger.info(f"  Number of trainable parameters = {per_device_trainable_numel} (per device)")
        if self.args.use_hybrid_parallel:
            # todo fix for pipeline_parallel_degree
            parts_num = max(self.args.tensor_parallel_degree, 1) * max(self.args.pipeline_parallel_degree, 1)
            if parts_num > 1:
                trainable_numel_tensor = paddle.to_tensor(per_device_trainable_numel, dtype="int64")
                paddle.distributed.all_reduce(trainable_numel_tensor)
                trainable_numel = trainable_numel_tensor.item() // self.args.dataset_world_size
                # the numel is roughly, because the tensor parallel still hold own bias or layer_norm weight without splited
                # so, the trainable numel is a little bigger than real.
                logger.info(f"  Number of trainable parameters = {trainable_numel} (all devices, roughly)")

        start_time = time.time()
        self._globalstep_last_start_time = time.time()
        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")
            if not args.ignore_data_skip:
                if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader.batch_sampler, NlpDistributedBatchSampler
                ):
                    consumed_samples = (
                        self.state.global_step
                        * args.train_batch_size
                        * args.gradient_accumulation_steps
                        * args.dataset_world_size
                    )
                    train_dataloader.batch_sampler.set_epoch(consumed_samples=consumed_samples)
                    logger.info(f"Set DistributedBatchSampler consumed_samples to {consumed_samples}")

        epoch_iterator = train_dataloader

        # steps_in_epoch = len(epoch_iterator)
        steps_in_epoch = (
            len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
        )

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        self.state.max_steps = int(max_steps)
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        tr_loss = paddle.to_tensor(0.0)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        if self.args.device == "npu" and self.args.flatten_param_grads:
            from .plugins.npu_plugin import npu_accelerate_plugin

            npu_accelerate_plugin(self.optimizer)
    
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                train_dataloader.batch_sampler, DistributedBatchSampler
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)

            step = -1
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            accum_logits = paddle.to_tensor(0.0)
            tr_loss = paddle.to_tensor(0.0)
            
            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                # for paddlenlp.utils.batch_sampler.DistributedBatchSampler
                # We use consumed_samples to reset the status
                
                # print(inputs)
                # breakpoint()
                
                if isinstance(train_dataloader, paddle.io.DataLoader) and isinstance(
                    train_dataloader.batch_sampler, NlpDistributedBatchSampler
                ):
                    if step == 0:
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(steps_trained_in_current_epoch)
                            steps_trained_progress_bar.close()
                            steps_trained_progress_bar = None
                        self._load_rng_state(resume_from_checkpoint)
                    step += steps_trained_in_current_epoch
                elif steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                dp_enabled = (
                    self.args.data_parallel_degree > 1 if self.args.use_hybrid_parallel else args.local_rank != -1
                )
                forbidden_no_sync = False
                # stage2 and stage3 should not no_sync, because the is no DDP wrapper and  no_sync API
                if self.sharding and (ShardingOption.SHARD_OP not in self.args.sharding):
                    forbidden_no_sync = True
                # hybrid_parallel (tp or pp) should not no_sync
                if self.args.use_hybrid_parallel and (
                    self.args.tensor_parallel_degree > 1 or self.args.pipeline_parallel_degree > 1
                ):
                    forbidden_no_sync = True

                availiable_no_sync = dp_enabled and not forbidden_no_sync

                is_no_sync = (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and availiable_no_sync
                    and args._no_sync_in_gradient_accumulation
                ) or (args.recompute and availiable_no_sync)
                # sharding
                # stage1. the same as ddp
                # stage2. manualy collect gradient on dp group

                model.train()
                inputs = self._prepare_inputs(inputs)
                labels = inputs.pop("labels")
                
                # Model forward
                if is_no_sync:
                # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        logits = self.compute_forward(model, inputs)
                else:
                    logits = self.compute_forward(model, inputs)

                accum_logits = paddle.add(accum_logits, logits.sum(axis=0, keepdim=True))
                
                # Compute loss
                if inputs["nth_chunk"][-1] == inputs["num_chunks"][-1]:

                    accum_logits = accum_logits / inputs["num_chunks"][-1]
                    with self.autocast_smart_context_manager():
                        loss = self.compute_loss(labels, logits=accum_logits, is_train=model.training)

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    if self.do_grad_scaling:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Reset accumulators
                    accum_logits = paddle.to_tensor(0.0)
                
                else:
                    logits.detach()
                    continue

                tr_loss += loss.detach()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Maunally collect gradients when group_sharded_parallel can't accept dp_group
                    # Case 1: Use sharding stage 2/3 with dp
                    # Case 2: Use recompute and dp
                    # local_rank != -1 don't means dp in networks.
                    if self.sharding and ShardingOption.SHARD_OP not in self.args.sharding:
                        if self.args.data_parallel_degree > 1 and not is_dp_group_support_in_group_sharded_parallel():
                            fused_allreduce_gradients(model.parameters(), fleet.get_hybrid_communicate_group())
                            if ShardingOption.FULL_SHARD in self.args.sharding:
                                # Why need sync on parm again ?
                                # TODO: fix this.
                                for p in model.parameters():
                                    if hasattr(p, "bw_storage"):
                                        assert p.grad is None, "This case shouldn't happen."
                                        p.bw_storage.scale_(1.0 / self.dp_group.nranks)
                                        paddle.distributed.all_reduce(p.bw_storage, group=self.dp_group)

                    # Case 2: Use recompute and dp / sharding stage1,
                    # manualy collect gradient for dp.
                    elif args.recompute and availiable_no_sync:
                        fused_allreduce_gradients(list(model.parameters()), None)

                    # pipeline parallel mode,  handle gradient merge here
                    if args.pipeline_parallel_degree > 1:
                        real_accumulate_steps = args.gradient_accumulation_steps
                        if hasattr(model, "_delay_scale_loss"):
                            if model._delay_scale_loss:
                                real_accumulate_steps *= model.accumulate_steps

                        for p in model._layers.parameters():
                            if hasattr(p, "main_grad") and p.main_grad is not None:
                                assert p.grad is None
                                p.main_grad = p.main_grad.scale(1.0 / real_accumulate_steps)
                            elif p.grad is not None:
                                p.grad = p.grad.scale(1.0 / real_accumulate_steps)

                    # Optimizer step
                    optimizer_was_run = True
                    if self.do_grad_scaling:
                        scale_before = self.scaler._scale.numpy()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler._scale.numpy()
                        optimizer_was_run = not self.scaler._cache_founf_inf
                        if not optimizer_was_run:
                            logger.warning(
                                f"optimizer not run, scale_before: {scale_before[0]}, scale_after: {scale_after[0]}"
                            )
                    else:
                        self.optimizer.step()

                    if optimizer_was_run:
                        self.lr_scheduler.step()

                    self.optimizer.clear_grad()

                    self.state.global_step += inputs["num_chunks"][-1].item()
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch

                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    
                    self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            if step < 0:
                logger.warning(
                    f"There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({self.state.max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, epoch, ignore_keys_for_eval, inputs=inputs)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\nTraining completed. \n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            if args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, LoRAModel):
                weight_name = LORA_WEIGHT_FILE_NAME
            elif isinstance(self.model, PrefixModelForCausalLM):
                weight_name = PREFIX_WEIGHT_FILE_NAME
            else:
                weight_name = PADDLE_WEIGHT_FILE_NAME
            best_model_path = os.path.join(
                self.state.best_model_checkpoint, _add_variant(weight_name, self.args.weight_name_suffix)
            )
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = paddle.load(best_model_path, return_numpy=True)
                # If the model is on the GPU, it still works!
                self._set_state_dict_in_model(state_dict)
            else:
                logger.warning(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

        self._total_loss_scalar += tr_loss.item()
        self.state.global_step = 1
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        metrics["train_loss"] = train_loss
        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not
                accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.dataset_world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_eval_iters: Optional[int] = -1,
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        if self.args.pipeline_parallel_degree > 1:
            # Only accept wrapped model for pipeline_parallel mode
            model = self.model_wrapped
        else:
            model = self.model

        if isinstance(dataloader, paddle.io.DataLoader):
            batch_size = dataloader.batch_sampler.batch_size
        elif isinstance(dataloader, paddle.fluid.dataloader.dataloader_iter._DataLoaderIterBase):
            # support for inner dataloader
            batch_size = dataloader._batch_sampler.batch_size
            # alias for inner dataloader
            dataloader.dataset = dataloader._dataset
        else:
            raise ValueError("Only support for paddle.io.DataLoader")

        num_samples = None
        if max_eval_iters > 0:
            # on eval limit steps
            num_samples = batch_size * self.args.dataset_world_size * max_eval_iters
            if isinstance(dataloader, paddle.fluid.dataloader.dataloader_iter._DataLoaderIterBase) and isinstance(
                dataloader._batch_sampler, NlpDistributedBatchSampler
            ):
                consumed_samples = (
                    ((self.state.global_step) // args.eval_steps)
                    * max_eval_iters
                    * args.per_device_eval_batch_size
                    * args.dataset_world_size
                )
                dataloader._batch_sampler.set_epoch(consumed_samples=consumed_samples)

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            if max_eval_iters > 0:
                logger.info(f"  Total prediction steps = {max_eval_iters}")
            else:
                logger.info(f"  Total prediction steps = {len(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
            if max_eval_iters > 0:
                logger.info(f"  Total prediction steps = {max_eval_iters}")

        logger.info(f"  Pre device batch size = {batch_size}")
        logger.info(f"  Total Batch size = {batch_size * self.args.dataset_world_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        losses = []
        # tmp_batch = None
        accum_logits = paddle.to_tensor(0.0)

        for step, inputs in enumerate(dataloader):

            labels = inputs.pop("labels")
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Model forward
            if self.args.pipeline_parallel_degree > 1:
            # hack for pipeline mode
                return self.prediction_pipeline_step(model, inputs, prediction_loss_only, ignore_keys)

            has_labels = all(inputs.get(k) is not None for k in self.label_names)
            
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            if has_labels:
                labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            with paddle.no_grad():
                with self.autocast_smart_context_manager():
                    logits = self.compute_forward(model, inputs)
                    accum_logits = paddle.add(accum_logits, logits.sum(axis=0, keepdim=True))                

            # Compute loss
            if inputs["nth_chunk"][-1] == inputs["num_chunks"][-1]:
                accum_logits = nested_detach(accum_logits)
                if isinstance(accum_logits, (list, tuple)) and len(accum_logits) == 1:
                    accum_logits = accum_logits[0]

                accum_logits = accum_logits / inputs["num_chunks"][-1]

                with paddle.no_grad():
                    with self.autocast_smart_context_manager():
                        loss = self.compute_loss(labels, logits=accum_logits, is_train=model.training)
                    loss = loss.mean().detach()

                logits = accum_logits
                labels = labels[-1, :].unsqueeze(axis=0)

                # Reset accumulators
                accum_logits = paddle.to_tensor(0.0)
            
            else:
                logits.detach()
                continue
                    
            # Update containers on host
            if loss is not None:
                # losses = self._nested_gather(loss.repeat(batch_size))
                losses = self._nested_gather(paddle.tile(loss, repeat_times=[batch_size, 1]))
                losses_host = losses if losses_host is None else paddle.concat((losses_host, losses), axis=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)

                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

            if max_eval_iters > 0 and step >= max_eval_iters - 1:
                break

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if num_samples is not None:
            pass
        elif has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        model.train()
        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.
        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)
        <Tip>
        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.
        </Tip>
        Returns: *NamedTuple* A namedtuple with the following keys:
            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.per_device_eval_batch_size * self.args.dataset_world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)
    

def main():
    # Parse the arguments.
    parser = PdArgumentParser((ModelArguments, DataArguments, PromptTuningArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Load the pretrained language model.
    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # text = tokenizer.convert_ids_to_tokens([1, 17416, 19509, 1397, 19574, 31, 58, 72, 245, 119, 104, 19676, 505, 1079, 19619, 3930, 17, 130, 1397, 19676, 436, 131, 4552, 9490, 19505, 250, 612, 338, 2763, 12456, 171, 612, 17555, 19660, 992, 204, 19748, 20011, 140, 38, 8, 19588, 826, 3586, 28, 517, 250, 612, 196, 171, 612, 19479, 603, 19719, 755, 487, 259, 4, 160, 200, 1342, 104, 912, 19578, 119, 104, 19748, 20011, 19556, 323, 1420, 19587, 40, 19465, 15012, 755, 19977, 19927, 12052, 276, 124, 12053, 104, 259, 4, 19480, 89, 245, 1342, 104, 911, 1405, 91, 728, 798, 152, 19472, 4, 89, 245, 1789, 119, 19466, 3930, 17, 768, 136, 1900, 139, 545, 19782, 19951, 19561, 19680, 19538, 4, 19469, 1056, 19564, 41, 392, 718, 5, 41, 503, 9, 3, 2])
    # print(text, len(text), text2, text3, text4)


    # Define the template for preprocess and the verbalizer for postprocess.
    template = AutoTemplate.create_from(
                                        data_args.prompt,
                                        # prompt="{'text': 'text_a'},{'hard': '這句話要包含的要素有'},{'mask': None, 'length': 1}",
                                        tokenizer=tokenizer, 
                                        max_length=training_args.max_seq_length, 
                                        model=model
                                        )
    logger.info("Using template: {}".format(template.prompt))

    label_file = os.path.join(data_args.data_dir, "label.txt")
    with open(label_file, "r", encoding="utf-8") as fp:
        label_words = defaultdict(list)
        for line in fp:
            data = line.strip().split("==")
            word = data[1] if len(data) > 1 else data[0].split("##")[-1]
            label_words[data[0]].append(word)
    verbalizer = SoftVerbalizer(label_words, tokenizer, model)

    # Load the few-shot datasets.
    train_ds, dev_ds, test_ds = load_local_dataset(
        data_path=data_args.data_dir, splits=["train", "dev", "test"], label_list=verbalizer.labels_to_ids, train_batch_size=training_args.train_batch_size , \
        eval_batch_size=training_args.eval_batch_size , chunk_len=training_args.max_seq_length, overlap_length=10,  other_tokens_length=3
    )


    # Define the criterion.
    criterion = paddle.nn.BCEWithLogitsLoss()

    # Initialize the prompt model with the above variables.
    prompt_model = PromptModelForSequenceClassification(
        model, template, verbalizer, freeze_plm=training_args.freeze_plm, freeze_dropout=training_args.freeze_dropout, 
    )

    # Define the metric function.
    def compute_metrics(eval_preds):
        metric = MetricReport()
        preds = F.sigmoid(paddle.to_tensor(eval_preds.predictions))
        metric.update(preds, paddle.to_tensor(eval_preds.label_ids))
        micro_f1_score, macro_f1_score, accuracy, precision, recall = metric.accumulate()
        return {"micro_f1_score": micro_f1_score, "macro_f1_score": macro_f1_score, "accuracy_score": accuracy, "precision_score": precision, "recall_score": recall}

    # Deine the early-stopping callback.
    callbacks = [EarlyStoppingCallback(early_stopping_patience=4, early_stopping_threshold=0.0)]
    # callbacks[0].on_prediction_step()

    # Initialize the trainer.
    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=PromptDataCollatorWithPadding(tokenizer, padding=True, return_tensors="pd"),
        criterion=criterion,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    # Training.
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Prediction.
    if training_args.do_predict:
        test_ret = trainer.predict(test_ds)
        trainer.log_metrics("test", test_ret.metrics)

    # Export static model.
    if training_args.do_export:
        export_path = os.path.join(training_args.output_dir, "export")
        trainer.export_model(export_path, export_type=model_args.export_type)


if __name__ == "__main__":
    main()
