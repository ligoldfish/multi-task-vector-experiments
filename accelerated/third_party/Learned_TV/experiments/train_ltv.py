import numpy as np
from copy import deepcopy
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import random
random.seed(2024)
import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import argparse

from datasets import load_dataset, concatenate_datasets
from utils.utils import *
from utils.IH_PTH import *
from utils.data import *
from utils.get_icl import *
from utils.hidden_states import *
from utils.model import *


def train_ltv(llm_name, device='cuda', dataset_name=None, st=None, ed=None, no_scale=False):
    model, tokenizer, embed, layers, final_ln, configuration, vocab_size, num_layer, d_model, num_heads, d_head = get_model(
        llm_name, half_precision=True
    )
    model.to(device)
    model.gradient_checkpointing_enable()
    if dataset_name is not None:
        dataset_names = [dataset_name]
    else:
        dataset_names = ['sst-2', 'trec', 'rte', 'snli', 'capital', 'capitalize', 'antonym']
    for dataset_name in dataset_names:
        num_class = num_class_dict[dataset_name]
        offset = offset_dict[llm_name]
        offset_label = offset + 1 if 'llama2' in model.name_or_path or 'llama-2' in model.name_or_path.lower() else offset

        demo_size = None

        if dataset_name in dataset_dict['classification']:
            dataset, df, prompt_dataset, prompt_df = get_classification_dataset(
                dataset_name, num_class, demo_size, seed=random.randint(0, 10000)
            )
        elif dataset_name in dataset_dict['nli']:
            dataset, df, prompt_dataset, prompt_df = get_nli_dataset(dataset_name, num_class, demo_size)
        elif dataset_name in dataset_dict['qa']:
            dataset, df, prompt_dataset, prompt_df = get_qa_dataset(dataset_name, num_class, demo_size)

        natural_mapping, random_mapping, flip_mapping, natural_labels, random_labels, flip_labels, \
        natural_labels_indices, random_labels_indices, flip_labels_indices, indices = \
            get_mapping_labels_indices(
                dataset_name,
                tokenizer,
                num_class,
                offset,
                concatenate_datasets([dataset, prompt_dataset])
            )

        source_folder_name = generate_folder_name(dataset_name, llm_name, 8, 'normal', '', 'source', 0)
        train_set = torch.load(f'{source_folder_name}/train_set.pt', weights_only=False)
        val_set = torch.load(f'{source_folder_name}/val_set.pt', weights_only=False)
        test_set = torch.load(f'{source_folder_name}/test_set.pt', weights_only=False)

        from torch.optim import AdamW
        from transformers import get_scheduler

        num_epochs = 10
        icl_num = 1
        save_flag = True

        score_folder_name = generate_folder_name(dataset_name, llm_name, icl_num, 'normal', '', 'score', 0)
        logs_folder_name = generate_folder_name(dataset_name, llm_name, icl_num, 'normal', '', 'logs', 0)
        tv_folder_name = generate_folder_name(dataset_name, llm_name, icl_num, 'normal', '', 'tv', 0)

        create_folder(score_folder_name)
        create_folder(logs_folder_name)
        create_folder(tv_folder_name)

        f = open(f'{logs_folder_name}/train_LTV.txt', 'w')

        def log_message(msg, tensor=None, tensor_path=None):
            f.write(msg + '\n')
            if tensor is not None and tensor_path:
                torch.save(tensor, tensor_path)

        early_stop_patience = 2
        best_acc_icl_mod = 0
        no_improvement_count = 0
        pos_to_intervene = -1

        if st is None and ed is None:
            st=0
            ed=num_layer
        for i in range(st, ed):

            inject_layers = [i]
            coef_file_name = f'{tv_folder_name}/{inject_layers}_{pos_to_intervene}.pt'

            if os.path.exists(coef_file_name):
                train_flag = False
                coef = torch.load(coef_file_name)
            else:
                train_flag = True
                coef = torch.nn.Parameter(
                    torch.empty((len(inject_layers), icl_num, d_model),
                                dtype=model.dtype, device=device).uniform_(-0.1, 0.1)
                )

                optimizer = AdamW([coef], lr=5e-3, weight_decay=0.01)
                lr_scheduler = get_scheduler(
                    "linear",
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=num_epochs * len(train_set),
                )

            if train_flag:
                for epoch in range(num_epochs):
                    acc_icl, acc_icl_mod = [], []
                    evaluate_set(
                        train_set.select(random.sample(range(len(train_set)), 100)),
                        model, tokenizer, natural_mapping,
                        model.dtype, num_layer, num_heads, llm_name,
                        offset_label, inject_layers, pos_to_intervene,
                        acc_icl, acc_icl_mod, coef, final_ln,
                        optimizer=optimizer, lr_scheduler=lr_scheduler, training=True
                    )

                    log_message(f'Epoch {epoch+1}, train; Before: {np.mean(acc_icl)}, After: {np.mean(acc_icl_mod)}')

                    acc_icl, acc_icl_mod = [], []
                    evaluate_set(
                        val_set, model, tokenizer, natural_mapping,
                        model.dtype, num_layer, num_heads, llm_name,
                        offset_label, inject_layers, pos_to_intervene,
                        acc_icl, acc_icl_mod, coef, final_ln
                    )

                    val_acc_mod = np.mean(acc_icl_mod)
                    log_message(f'Epoch {epoch+1}, val; Before: {np.mean(acc_icl)}, After: {val_acc_mod}')

                    if val_acc_mod > best_acc_icl_mod or epoch == 0:
                        best_acc_icl_mod = val_acc_mod
                        no_improvement_count = 0
                        best_coef = deepcopy(coef)
                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= early_stop_patience:
                            log_message(f"Early stopping at epoch {epoch+1}")
                            final_coef = best_coef
                            if save_flag:
                                create_folder(f'coefs/{dataset_name}/{llm_name}/')
                                torch.save(final_coef, coef_file_name)
                            break
                        if epoch == num_epochs - 1:
                            final_coef = best_coef
                            if save_flag:
                                create_folder(f'coefs/{dataset_name}/{llm_name}/')
                                torch.save(final_coef, coef_file_name)
            else:
                final_coef = coef

            acc_icl, acc_icl_mod = [], []
            evaluate_set(
                test_set, model, tokenizer, natural_mapping,
                model.dtype, num_layer, num_heads, llm_name,
                offset_label, inject_layers, pos_to_intervene,
                acc_icl, acc_icl_mod, final_coef, final_ln
            )

            test_acc_mod = np.mean(acc_icl_mod)
            log_message(f'Test ICL Acc: {np.mean(acc_icl)}, After: {test_acc_mod}')
            torch.save(test_acc_mod, f'{score_folder_name}/{inject_layers}_{pos_to_intervene}.pt')
        
        if not no_scale:
            inject_layers = [int(num_layer / 2)]

            for i in range(6):
                if i < 5:
                    pos_to_intervene = -1
                    coef_file_name = f'{tv_folder_name}/{inject_layers}_{pos_to_intervene}_{i+1}.pt'
                else:
                    pos_to_intervene = 3
                    coef_file_name = f'{tv_folder_name}/{inject_layers}_{pos_to_intervene}.pt'

                if os.path.exists(coef_file_name):
                    train_flag = False
                    coef = torch.load(coef_file_name).to(device)
                else:
                    train_flag = True
                    coef = torch.nn.Parameter(
                        torch.empty((len(inject_layers), icl_num, d_model),
                                    dtype=model.dtype, device=device).uniform_(-0.1, 0.1)
                    )

                    optimizer = AdamW([coef], lr=5e-3, weight_decay=0.01)
                    lr_scheduler = get_scheduler(
                        "linear",
                        optimizer=optimizer,
                        num_warmup_steps=0,
                        num_training_steps=num_epochs * len(train_set),
                    )

                if train_flag:
                    for epoch in range(num_epochs):
                        acc_icl, acc_icl_mod = [], []
                        evaluate_set(
                            train_set.select(random.sample(range(len(train_set)), 100)),
                            model, tokenizer, natural_mapping,
                            model.dtype, num_layer, num_heads, llm_name,
                            offset_label, inject_layers, pos_to_intervene,
                            acc_icl, acc_icl_mod, coef, final_ln,
                            optimizer=optimizer, lr_scheduler=lr_scheduler, training=True
                        )

                        log_message(f'Epoch {epoch+1}, train; Before: {np.mean(acc_icl)}, After: {np.mean(acc_icl_mod)}')

                        acc_icl, acc_icl_mod = [], []
                        evaluate_set(
                            val_set, model, tokenizer, natural_mapping,
                            model.dtype, num_layer, num_heads, llm_name,
                            offset_label, inject_layers, pos_to_intervene,
                            acc_icl, acc_icl_mod, coef, final_ln
                        )

                        val_acc_mod = np.mean(acc_icl_mod)
                        log_message(f'Epoch {epoch+1}, val; Before: {np.mean(acc_icl)}, After: {val_acc_mod}')

                        if val_acc_mod > best_acc_icl_mod or epoch == 0:
                            best_acc_icl_mod = val_acc_mod
                            no_improvement_count = 0
                            best_coef = deepcopy(coef)
                        else:
                            no_improvement_count += 1
                            if no_improvement_count >= early_stop_patience:
                                log_message(f"Early stopping at epoch {epoch+1}")
                                final_coef = best_coef
                                if save_flag:
                                    create_folder(f'coefs/{dataset_name}/{llm_name}/')
                                    torch.save(final_coef, coef_file_name)
                                break
                            if epoch == num_epochs - 1:
                                final_coef = best_coef
                                if save_flag:
                                    create_folder(f'coefs/{dataset_name}/{llm_name}/')
                                    torch.save(final_coef, coef_file_name)
                else:
                    final_coef = coef

                acc_icl, acc_icl_mod = [], []
                evaluate_set(
                    test_set, model, tokenizer, natural_mapping,
                    model.dtype, num_layer, num_heads, llm_name,
                    offset_label, inject_layers, pos_to_intervene,
                    acc_icl, acc_icl_mod, final_coef, final_ln
                )

                test_acc_mod = np.mean(acc_icl_mod)
                log_message(f'Test ICL Acc: {np.mean(acc_icl)}, After: {test_acc_mod}')

                if i == 0 or i == 5:
                    torch.save(test_acc_mod, f'{score_folder_name}/{inject_layers}_{pos_to_intervene}.pt')

        f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_name", required=True, type=str)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--dataset_name", default=None, type=str, help="Name of the dataset for training LTV")
    parser.add_argument("--st", default=None, type=int, help="Start layer for training")
    parser.add_argument("--ed", default=None, type=int, help="End layer for training")
    parser.add_argument("--no_scale", action='store_true', help="Whether to skip training for scale analysis")
    args = parser.parse_args()

    train_ltv(args.llm_name, args.device, args.dataset_name, args.st, args.ed, args.no_scale)


if __name__ == "__main__":
    main()