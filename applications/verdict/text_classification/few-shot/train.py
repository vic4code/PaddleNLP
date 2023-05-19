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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# yapf: disable
@dataclass
class DataArguments:
    data_dir: str = field(default="./data", metadata={"help": "The dataset dictionary includes train.txt, dev.txt and label.txt files."})
    prompt: str = field(default=None, metadata={"help": "The input prompt for tuning."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-3.0-base-zh", metadata={"help": "The build-in pretrained model or the path to local model."})
    export_type: str = field(default='paddle', metadata={"help": "The type to export. Support `paddle` and `onnx`."})
# yapf: enable


class PromptTrainer(PromptTrainer):
    def chunking(self, dataset, max_length=self.args.max_length, prompt=self.args.prompt, other_tokens_length=4):
        
        sequence = dataset['text_a']
        sequence_length = len(sequence)
        divider = max_length - len(prompt) - other_tokens_length
        num_chunks = sequence_length // divider
        chunked_dataset = []

        for i in range(num_chunks):
            chunked_dataset.append(sequence[i:i + i ])
        

        chunked_dataset = dataset
        return chunked_dataset
    

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the total loss for every batch.
        """
        if "labels" not in inputs:
            raise ValueError("Fail to compute loss as `labels` not in {}.".format(inputs))
        labels = inputs["labels"]

        input_dict = inputs.copy()

        sequence_len = input_dict['input_ids'].shape[-1]
        # input_ids[:, ]
        # inpu
        # print(input_dict, input_dict.keys())
        # return

        if self.criterion is not None:
            # pop labels to move loss computation out of the model
            input_dict.pop("labels")
            input_dict["return_hidden_states"] = True
            logits, hidden_states = model(**input_dict)
            loss = self.criterion(logits, labels)

            if self.args.use_rdrop:
                loss = self._compute_rdrop_loss(model, input_dict, labels, logits, loss)

            if self.args.use_rgl:
                loss += self._compute_rgl_loss(hidden_states, labels)
        else:
            loss, logits = model(**input_dict)

        outputs = (loss, logits)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Layer, inputs: Dict[str, Union[paddle.Tensor, Any]]) -> paddle.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to train.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `paddle.Tensor`: The tensor with training loss on this batch.
        """
        if self.args.pipeline_parallel_degree > 1:
            return self.training_pipeline_step(model, inputs)

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()
    
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

        print(self.train_dataset.__getitem__(0), len(self.train_dataset.__getitem__(0)['input_ids']))

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
        print(dir(train_dataloader))

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

            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                # for paddlenlp.utils.batch_sampler.DistributedBatchSampler
                # We use consumed_samples to reset the status
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

                if is_no_sync:
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                tr_loss += tr_loss_step

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

                    self.state.global_step += 1
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
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)

        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)


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

    text = tokenizer.convert_ids_to_tokens([1, 17416, 19509, 1397, 19574, 31, 58, 72, 245, 119, 104, 19676, 505, 1079, 19619, 3930, 17, 130, 1397, 19676, 436, 131, 4552, 9490, 19505, 250, 612, 338, 2763, 12456, 171, 612, 17555, 19660, 992, 204, 19748, 20011, 140, 38, 8, 19588, 826, 3586, 28, 517, 250, 612, 196, 171, 612, 19479, 603, 19719, 755, 487, 259, 4, 160, 200, 1342, 104, 912, 19578, 119, 104, 19748, 20011, 19556, 323, 1420, 19587, 40, 19465, 15012, 755, 19977, 19927, 12052, 276, 124, 12053, 104, 259, 4, 19480, 89, 245, 1342, 104, 911, 1405, 91, 728, 798, 152, 19472, 4, 89, 245, 1789, 119, 19466, 3930, 17, 768, 136, 1900, 139, 545, 19782, 19951, 19561, 19680, 19538, 4, 19469, 1056, 19564, 41, 392, 718, 5, 41, 503, 9, 3, 2])
    print(text, len(text))

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
        data_path=data_args.data_dir, splits=["train", "dev", "test"], label_list=verbalizer.labels_to_ids
    )



    # print(isinstance(train_ds, paddle.io.IterableDataset))
    # # print(is_train_ds.__getitem__(0))
    # return

    # Define the criterion.
    criterion = paddle.nn.BCEWithLogitsLoss()

    # Initialize the prompt model with the above variables.
    prompt_model = PromptModelForSequenceClassification(
        model, template, verbalizer, freeze_plm=training_args.freeze_plm, freeze_dropout=training_args.freeze_dropout
    )

    # Define the metric function.
    def compute_metrics(eval_preds):
        metric = MetricReport()
        preds = F.sigmoid(paddle.to_tensor(eval_preds.predictions))
        metric.update(preds, paddle.to_tensor(eval_preds.label_ids))
        micro_f1_score, macro_f1_score = metric.accumulate()
        return {"micro_f1_score": micro_f1_score, "macro_f1_score": macro_f1_score}

    # Deine the early-stopping callback.
    callbacks = [EarlyStoppingCallback(early_stopping_patience=4, early_stopping_threshold=0.0)]


    # print(training_args)

    # return 
    # Initialize the trainer.
    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=training_args,
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
